"""LoRA / QLoRA fine-tuning on the dataset built by data.py.

    uv run python train.py --run_name my-run
    uv run python train.py --run_name my-run --resume   # continue from the latest checkpoint

Everything for a run lives in runs/<run_name>/: checkpoints, the final adapter,
and (with --merge) a standalone merged model. Defaults target a ~40 GB GPU with
QLoRA; see the README for other GPU sizes.
"""

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig
from transformers import HfArgumentParser
from trl import SFTConfig, SFTTrainer

import model_spec
from data import DATASET_FORMAT_VERSION


@dataclass
class TrainConfig:
    dataset_dir: str = "data/dataset"
    run_name: str | None = field(default=None, metadata={"help": "Defaults to a timestamp."})
    runs_dir: str = "runs"
    resume: bool = field(default=False, metadata={"help": "Resume --run_name from its latest checkpoint."})
    quantization: model_spec.Quantization = field(
        default="4bit", metadata={"help": '"4bit" (QLoRA) fits ~24-40 GB; "none" (bf16 LoRA) wants ~80 GB.'}
    )

    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    learning_rate: float = 1e-4
    num_train_epochs: float = 1.0
    max_steps: int = field(default=-1, metadata={"help": "Overrides num_train_epochs if > 0."})
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 32
    per_device_eval_batch_size: int = 1
    max_eval_chunks: int = field(default=200, metadata={"help": "Random subset of the eval split used for eval loss."})
    # Values < 1 are fractions of the total number of optimizer steps, so every run
    # gets ~20 evals and ~10 checkpoints regardless of dataset size.
    eval_steps: float = 0.05
    save_steps: float = 0.1
    save_total_limit: int | None = None
    logging_steps: int = 5
    seed: int = 42

    merge: bool = field(default=False, metadata={"help": "Also save a merged bf16 model (needs ~24 GB CPU RAM)."})


def latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    checkpoints = [d for d in checkpoint_dir.glob("checkpoint-*") if d.name.split("-")[-1].isdigit()]
    return max(checkpoints, key=lambda d: int(d.name.split("-")[-1]), default=None)


def read_dataset_meta(dataset_dir: Path) -> dict:
    meta_path = dataset_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found. Build the dataset with data.py first.")
    meta = json.loads(meta_path.read_text())
    if meta.get("format_version") != DATASET_FORMAT_VERSION:
        raise RuntimeError(
            f"Dataset format version {meta.get('format_version')} != {DATASET_FORMAT_VERSION}; rebuild it with data.py."
        )
    return meta


def check_dataset_tokenizer(meta: dict, tokenizer) -> None:
    if meta["tokenizer_fingerprint"] != model_spec.tokenizer_fingerprint(tokenizer):
        raise RuntimeError(
            f"The dataset was tokenized for {meta['model_name']} with a different tokenizer; rebuild it with data.py."
        )


def check_loss_parity(trainer: SFTTrainer, sample_ids: list[int], tolerance: float = 1e-3) -> None:
    """TRL may patch the model's forward to compute the loss itself (e.g. loss_type="chunked_nll"),
    re-implementing model details such as Gemma's final logit softcapping. Verify that the loss it
    optimizes equals the loss computed from the model's own logits."""
    model = trainer.model
    was_training = model.training
    model.eval()  # disable LoRA dropout
    input_ids = torch.tensor([sample_ids], device=trainer.args.device)
    with torch.no_grad():
        trainer_loss = model(input_ids=input_ids, labels=input_ids).loss.float()
        logits = model(input_ids=input_ids).logits.float()
        reference_loss = torch.nn.functional.cross_entropy(logits[0, :-1], input_ids[0, 1:])
    model.train(was_training)
    if not torch.isclose(trainer_loss, reference_loss, rtol=tolerance, atol=tolerance):
        raise RuntimeError(
            f"Training loss {trainer_loss.item():.5f} != reference loss {reference_loss.item():.5f}. The trainer's "
            'loss computation doesn\'t match the model (e.g. missing logit softcapping); try loss_type="nll".'
        )
    print(f"Loss parity check passed ({trainer_loss.item():.4f} vs {reference_loss.item():.4f})")


def main():
    (cfg,) = HfArgumentParser(TrainConfig).parse_args_into_dataclasses()

    if cfg.resume and cfg.run_name is None:
        raise ValueError("--resume needs --run_name.")
    run_name = cfg.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(cfg.runs_dir) / run_name
    checkpoint_dir = run_dir / "checkpoints"

    resume_from = None
    if cfg.resume:
        resume_from = latest_checkpoint(checkpoint_dir)
        if resume_from is None:
            raise FileNotFoundError(f"No checkpoints in {checkpoint_dir} to resume from.")
    elif run_dir.exists():
        raise FileExistsError(f"{run_dir} already exists. Pass --resume to continue it, or pick another --run_name.")

    dataset_meta = read_dataset_meta(Path(cfg.dataset_dir))
    model_name = dataset_meta["model_name"]
    if resume_from is not None and model_spec.adapter_base_model(resume_from) != model_name:
        raise RuntimeError(
            f"{resume_from} was trained on {model_spec.adapter_base_model(resume_from)}, "
            f"but the dataset is for {model_name}."
        )

    processor = model_spec.load_processor(model_name)
    check_dataset_tokenizer(dataset_meta, processor.tokenizer)
    dataset = load_from_disk(cfg.dataset_dir)
    eval_dataset = dataset["eval"].shuffle(seed=cfg.seed)
    eval_dataset = eval_dataset.select(range(min(cfg.max_eval_chunks, len(eval_dataset))))
    if len(eval_dataset) == 0:
        raise RuntimeError("The dataset has no eval chunks; rebuild it with a larger --eval_fraction.")

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"train_config_{datetime.now():%Y-%m-%d_%H-%M-%S}.json").write_text(
        json.dumps({"train": asdict(cfg), "dataset": dataset_meta}, indent=2, ensure_ascii=False)
    )
    print(f"\033[1m{'Resuming from ' + str(resume_from) if resume_from else 'Starting run ' + str(run_dir)}\033[0m")

    model = model_spec.load_model(model_name, cfg.quantization)
    model.config.use_cache = False
    model_spec.check_lora_targets(model)

    peft_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=model_spec.LORA_TARGET_REGEX,
        task_type="CAUSAL_LM",
    )

    sft_config = SFTConfig(
        output_dir=str(checkpoint_dir),
        run_name=run_name,
        seed=cfg.seed,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=cfg.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,  # < 1: fraction of total steps
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        use_cpu=not torch.cuda.is_available(),  # only useful for smoke tests
        # The dataset is pre-tokenized (input_ids), so TRL skips tokenization and
        # EOS insertion. Chunks already fit max_length; loss is on every token.
        max_length=dataset_meta["max_length"],
        packing=False,  # packing would let unrelated channels attend to each other
        completion_only_loss=False,
        # Computes lm_head + cross-entropy in chunks: the full logits tensor for
        # Gemma's 262k vocab would otherwise dominate activation memory.
        loss_type="chunked_nll",
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        eval_on_start=resume_from is None,  # base model baseline (LoRA starts as a no-op)
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        # Pass the processor, not just the tokenizer: TRL then treats the model as
        # multimodal and reads final_logit_softcapping from text_config (with a
        # plain tokenizer, chunked_nll silently skips the softcapping).
        processing_class=processor,
    )
    trainer.model.print_trainable_parameters()
    check_loss_parity(trainer, eval_dataset[0]["input_ids"][:512])

    steps_per_epoch = math.ceil(
        len(dataset["train"]) / (cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps)
    )
    print(f"{len(dataset['train'])} train chunks, {steps_per_epoch} optimizer steps per epoch")

    trainer.train(resume_from_checkpoint=str(resume_from) if resume_from else None)

    adapter_dir = run_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    print(f"Saved LoRA adapter to {adapter_dir}")

    if cfg.merge:
        from peft import PeftModel

        del model, trainer
        torch.cuda.empty_cache()
        # Merging needs an unquantized base; load it on CPU so this works without a big GPU.
        base = model_spec.load_model(model_name, quantization="none", device_map="cpu")
        merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
        merged_dir = run_dir / "merged"
        merged.save_pretrained(str(merged_dir))
        processor.save_pretrained(str(merged_dir))
        print(f"Saved merged model to {merged_dir}")


if __name__ == "__main__":
    main()
