"""Score the checkpoints of a run on a held-out dataset and plot the loss curve.

    ./run.sh evaluate.py --run_dir runs/my-run --dataset_dir data/newer_chats

The base model and every checkpoint are evaluated on the same chunks, so the
numbers say how much the fine-tune actually learned about the channels. The most
honest dataset for this is chat from *after* the training data was exported
(build one with `data.py --eval_fraction 1`, which puts everything in the eval
split): unlike a random split, it can't share a conversation with the training
data.

Writes a Markdown table, a JSON file with all numbers, and a PNG plotting the
validation loss over the training loss curve.
"""

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import PeftModel
from tqdm import tqdm
from transformers import HfArgumentParser

import model_spec
from inference import load_for_inference, trained_context_tokens
from sample import adapter_name, collect_adapters
from train import check_dataset_tokenizer, read_dataset_meta


@dataclass
class EvalConfig:
    run_dir: str | None = field(
        default=None, metadata={"help": "Evaluate every checkpoint and the final adapter of a run."}
    )
    adapters: list[str] = field(
        default_factory=list, metadata={"help": "Explicit adapter dirs (instead of --run_dir)."}
    )
    include_base: bool = field(default=True, metadata={"help": "Also evaluate the base model (= step 0)."})
    dataset_dir: str = "data/dataset"
    split: str = "eval"
    max_chunks: int = field(default=0, metadata={"help": "0 = all; otherwise a random subset of that many chunks."})
    quantization: model_spec.Quantization | None = field(
        default=None, metadata={"help": 'Defaults to "4bit" on GPU; affects the losses slightly.'}
    )
    seed: int = 0
    output: str | None = field(default=None, metadata={"help": "Markdown path; .json/.png are written next to it."})
    plot: bool = True


def adapter_step(path: Path) -> int | None:
    """The optimizer step an adapter was saved at, for the x axis of the plot."""
    state = path / "trainer_state.json"  # saved in checkpoints, but not next to the final adapter
    if state.exists():
        return json.loads(state.read_text()).get("global_step")
    steps = [int(d.name.split("-")[-1]) for d in (path.parent / "checkpoints").glob("checkpoint-*")]
    return max(steps, default=None)  # the final adapter is the end of training


def training_curve(run_dir: Path) -> list[tuple[int, float]]:
    """(step, training loss) from the newest checkpoint's trainer state."""
    states = sorted((run_dir / "checkpoints").glob("checkpoint-*/trainer_state.json"), key=adapter_step_of_state)
    if not states:
        return []
    history = json.loads(states[-1].read_text())["log_history"]
    return [(entry["step"], entry["loss"]) for entry in history if "loss" in entry]


def adapter_step_of_state(state: Path) -> int:
    return int(state.parent.name.split("-")[-1])


def check_label_shift(model, ids: list[int], tolerance: float = 0.01) -> None:
    """The loss we report comes from the model's own forward with labels=input_ids, which is only
    the next-token loss if the model shifts the labels itself. Every causal LM in transformers does,
    but a model that didn't would silently report a much lower (and meaningless) loss here."""
    input_ids = torch.tensor([ids], device=model.device)
    with torch.no_grad():
        loss = model(input_ids=input_ids, labels=input_ids).loss.float()
        logits = model(input_ids=input_ids).logits.float()
        shifted = torch.nn.functional.cross_entropy(logits[0, :-1], input_ids[0, 1:])
        unshifted = torch.nn.functional.cross_entropy(logits[0], input_ids[0])
    del logits
    torch.cuda.empty_cache()
    if not torch.isclose(loss, shifted, rtol=tolerance, atol=0.0):
        raise RuntimeError(
            f"labels=input_ids gives loss {loss.item():.5f}, but next-token loss is {shifted.item():.5f} "
            f"(unshifted: {unshifted.item():.5f}). The model doesn't shift labels the way we assume."
        )


@torch.no_grad()
def dataset_loss(model, dataset, description: str) -> dict:
    """Mean loss per predicted token over the whole split, plus a per-channel breakdown."""
    per_chunk = []
    for row in tqdm(dataset, desc=description, leave=False):
        input_ids = torch.tensor([row["input_ids"]], device=model.device)
        output = model(input_ids=input_ids, labels=input_ids)
        predicted = input_ids[0, 1:]
        correct = (output.logits[0, :-1].argmax(-1) == predicted).sum().item()
        per_chunk.append(
            {
                "channel": row["channel"],
                "tokens": predicted.numel(),
                "loss": output.loss.float().item(),
                "correct": correct,
            }
        )
        del output
    torch.cuda.empty_cache()

    def aggregate(chunks: list[dict]) -> dict:
        tokens = sum(c["tokens"] for c in chunks)
        loss = sum(c["loss"] * c["tokens"] for c in chunks) / tokens
        return {
            "loss": loss,
            "perplexity": math.exp(loss),
            "token_accuracy": sum(c["correct"] for c in chunks) / tokens,
            "tokens": tokens,
            "chunks": len(chunks),
        }

    channels = sorted({c["channel"] for c in per_chunk})
    return {
        **aggregate(per_chunk),
        "per_channel": {channel: aggregate([c for c in per_chunk if c["channel"] == channel]) for channel in channels},
        "per_chunk": per_chunk,
    }


def smooth(values: list[float], factor: float = 0.05) -> list[float]:
    average, out = values[0], []
    for value in values:
        average += factor * (value - average)
        out.append(average)
    return out


def write_plot(results: list[dict], curve: list[tuple[int, float]], split: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")  # no display on a training box
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(9, 5.5))
    if curve:
        steps, losses = zip(*curve, strict=True)
        axes.plot(steps, losses, color="C0", alpha=0.2, linewidth=0.8, label="training loss (per step)")
        axes.plot(steps, smooth(list(losses)), color="C0", linewidth=1.8, label="training loss (smoothed)")
    points = sorted((r["step"], r["loss"]) for r in results if r["step"] is not None)
    if points:
        steps, losses = zip(*points, strict=True)
        axes.plot(steps, losses, "o-", color="C3", linewidth=1.8, label=f"validation loss ({split})")
    axes.set_xlabel("optimizer step")
    axes.set_ylabel("loss")
    axes.grid(alpha=0.3)
    axes.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    print(f"Wrote {path}")


def main():
    (cfg,) = HfArgumentParser(EvalConfig).parse_args_into_dataclasses()
    adapters = collect_adapters(cfg.run_dir, cfg.adapters, cfg.include_base)

    base_names = {model_spec.adapter_base_model(a) for a in adapters}
    if len(base_names) > 1:
        raise RuntimeError(f"Adapters were trained on different base models: {base_names}")
    base_name = base_names.pop() if base_names else model_spec.DEFAULT_MODEL_NAME

    dataset_meta = read_dataset_meta(Path(cfg.dataset_dir))
    dataset = load_from_disk(cfg.dataset_dir)
    if cfg.split not in dataset:
        raise ValueError(f'{cfg.dataset_dir} has no "{cfg.split}" split (it has {list(dataset)}); try --split train.')
    split = dataset[cfg.split]
    if cfg.max_chunks:
        split = split.shuffle(seed=cfg.seed).select(range(min(cfg.max_chunks, len(split))))
    trained = trained_context_tokens(adapters[0]) if adapters else None
    if trained is not None and dataset_meta["max_length"] > trained:
        print(
            f"Warning: the chunks are up to {dataset_meta['max_length']} tokens, but the run was trained on "
            f"{trained}; the loss then covers positions the model never saw during fine-tuning."
        )

    model, tokenizer = load_for_inference(base_name, cfg.quantization)
    check_dataset_tokenizer(dataset_meta, tokenizer)
    # (label, step, PEFT adapter name); adapter name None = base model without adapters.
    variants: list[tuple[str, int | None, str | None]] = [("base model", 0, None)] if cfg.include_base else []
    for adapter in adapters:
        name = adapter_name(adapter)
        if not isinstance(model, PeftModel):
            model = PeftModel.from_pretrained(model, str(adapter), adapter_name=name)
        else:
            model.load_adapter(str(adapter), adapter_name=name)
        variants.append((str(adapter), adapter_step(adapter), name))
    model.eval()
    model.config.use_cache = False

    check_label_shift(model, split[0]["input_ids"])

    results = []
    for label, step, name in variants:
        if name is None and isinstance(model, PeftModel):
            with model.disable_adapter():
                metrics = dataset_loss(model, split, label)
        else:
            if name is not None:
                model.set_adapter(name)
            metrics = dataset_loss(model, split, label)
        results.append({"label": label, "step": step, **metrics})
        print(
            f"{label}: loss {metrics['loss']:.4f}, perplexity {metrics['perplexity']:.2f}, "
            f"token accuracy {metrics['token_accuracy']:.3f}"
        )

    timestamp = datetime.now()
    default_output = Path(cfg.run_dir or ".") / f"eval_{cfg.split}_{timestamp:%Y-%m-%d_%H-%M-%S}.md"
    output = Path(cfg.output) if cfg.output else default_output
    channels = sorted({c for r in results for c in r["per_channel"]})
    header = ["step", "model", "loss", "perplexity", "token acc."] + [f"loss: {c}" for c in channels]
    lines = [
        f"# Validation on `{cfg.dataset_dir}` ({cfg.split} split) — {timestamp:%Y-%m-%d %H:%M}",
        "",
        f"{len(split)} chunks, {sum(r['tokens'] for r in results[:1])} predicted tokens, base model `{base_name}`, "
        f"quantization `{cfg.quantization or 'default'}`.",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for r in sorted(results, key=lambda r: (r["step"] is None, r["step"])):
        row = [
            str(r["step"] if r["step"] is not None else "-"),
            f"`{r['label']}`",
            f"{r['loss']:.4f}",
            f"{r['perplexity']:.2f}",
            f"{r['token_accuracy']:.3f}",
        ] + [f"{r['per_channel'][c]['loss']:.3f}" if c in r["per_channel"] else "-" for c in channels]
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "Per-channel chunks: " + ", ".join(
        f"{c} ({results[0]['per_channel'][c]['chunks']} chunks, {results[0]['per_channel'][c]['tokens']} tokens)"
        for c in channels
    ), ""]
    output.write_text("\n".join(lines))
    print(f"Wrote {output}")

    output.with_suffix(".json").write_text(
        json.dumps(
            {"dataset_dir": cfg.dataset_dir, "split": cfg.split, "base_model": base_name, "results": results},
            indent=2,
            ensure_ascii=False,
        )
    )
    if cfg.plot:
        curve = training_curve(Path(cfg.run_dir)) if cfg.run_dir else []
        write_plot(results, curve, cfg.split, output.with_suffix(".png"))


if __name__ == "__main__":
    main()
