"""QLoRA fine-tuning of a Qwen3.6-27B-class model on the ChatML chat dataset.

Run ``python data.py`` first to produce ``data/dataset.jsonl``, then::

    uv run python train.py

Tuned for a single 80 GB H100. Resumes automatically from the latest checkpoint
in ``checkpoints/`` if one exists.
"""

import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

import constants

# Linear sub-layers we attach LoRA to. We match by full module name and exclude
# the vision tower so adapters only touch the language model.
_PROJ_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
_VISION_MARKERS = ("visual", "vision", "merger", "patch_embed")


def latest_checkpoint(checkpoint_dir: Path):
    if not checkpoint_dir.exists():
        return None
    checkpoints = [d for d in checkpoint_dir.iterdir() if d.name.startswith("checkpoint-")]
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda d: int(d.name.split("-")[1]))


def language_linear_targets(model) -> list[str]:
    """Full names of the language-model Linear layers (vision tower excluded)."""
    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not name.endswith(_PROJ_SUFFIXES):
            continue
        if any(marker in name.lower() for marker in _VISION_MARKERS):
            continue
        targets.append(name)
    if not targets:
        raise RuntimeError("No language-model Linear layers found to target with LoRA.")
    return targets


def main():
    out_model_name = "engine_dev_model_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print("out_model_name:", out_model_name)

    checkpoint_dir = Path(constants.checkpoint_dir)
    resume = latest_checkpoint(checkpoint_dir)
    if resume:
        print(f"\033[1mResuming from {resume}\033[0m")
    else:
        print("\033[1mStarting training from scratch\033[0m")

    processor = AutoProcessor.from_pretrained(constants.model_name, token=constants.hf_token)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForMultimodalLM.from_pretrained(
        constants.model_name,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",  # works on both CUDA and ROCm, no extra deps
        token=constants.hf_token,
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # Scope LoRA to the language model only; leave the vision encoder frozen.
        target_modules=language_linear_targets(model),
    )

    dataset = load_dataset("json", data_files=constants.dataset_path, split="train")
    dataset = dataset.train_test_split(test_size=0.01, seed=42)

    sft_config = SFTConfig(
        output_dir=str(checkpoint_dir),
        num_train_epochs=2,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,  # effective batch size = 32
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        # Dataset is plain {"text": ...} language modeling -> full-sequence loss.
        dataset_text_field="text",
        completion_only_loss=False,
        # data.py already packs messages into channel-prefixed chunks near the
        # context limit, so we don't use TRL packing: it would concatenate
        # unrelated channels into one sequence and, under sdpa attention, let
        # them attend across boundaries (teaching artificial transitions). Each
        # row trains as one independent sequence instead.
        packing=False,
        max_length=constants.max_token_context_length,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=5,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    trainer.train(resume_from_checkpoint=str(resume) if resume else None)

    results = trainer.evaluate()
    print(f"Final eval loss: {results['eval_loss']:.4f}")

    adapter_path = f"./{out_model_name}_LORA"
    trainer.save_model(adapter_path)
    processor.save_pretrained(adapter_path)
    print(f"Saved LoRA adapter to {adapter_path}")

    # Optional: produce a standalone merged checkpoint. Merging must be done on
    # an un-quantized base, so this reloads the base in bf16 (~54 GB) — enable
    # only if you have the memory/need it for deployment.
    if os.environ.get("MERGE_MODEL") == "1":
        from peft import PeftModel

        del model, trainer
        torch.cuda.empty_cache()
        base = AutoModelForMultimodalLM.from_pretrained(
            constants.model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            token=constants.hf_token,
        )
        merged = PeftModel.from_pretrained(base, adapter_path).merge_and_unload()
        merged_path = f"./{out_model_name}_merged"
        merged.save_pretrained(merged_path)
        processor.save_pretrained(merged_path)
        print(f"Saved merged model to {merged_path}")

    print("Finished :D")


if __name__ == "__main__":
    main()
