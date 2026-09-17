"""Compare checkpoints (and the base model) by continuing held-out conversations.

    ./run.sh sample.py --run_dir runs/my-run

For a few eval chunks (the most recent messages of the largest channels), the
first --prompt_tokens tokens are used as the prompt, and every model continues
them for --messages messages with the same random seed. Loss alone says little
about whether the output reads like the real channel; this lets you judge that
side by side. The result is written as a Markdown file.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from datasets import load_from_disk
from peft import PeftModel
from transformers import HfArgumentParser, set_seed

import model_spec
from chat_format import ChatFormat
from generate import Conversation, SamplingConfig, load_for_inference, resolve_context_tokens, split_chunk


@dataclass
class SampleConfig:
    run_dir: str | None = field(
        default=None, metadata={"help": "Sample every checkpoint and the final adapter of a run."}
    )
    adapters: list[str] = field(
        default_factory=list, metadata={"help": "Explicit adapter dirs (instead of --run_dir)."}
    )
    include_base: bool = True
    dataset_dir: str = "data/dataset"
    num_prompts: int = 3
    prompt_tokens: int = field(default=512, metadata={"help": "Should leave room for --messages."})
    messages: int = 12
    quantization: model_spec.Quantization | None = None
    seed: int = 0
    output: str | None = field(default=None, metadata={"help": "Defaults to <run_dir>/samples_<time>.md."})


def collect_adapters(cfg: SampleConfig) -> list[Path]:
    adapters = [Path(a) for a in cfg.adapters]
    if cfg.run_dir is not None:
        run_dir = Path(cfg.run_dir)
        checkpoints = sorted((run_dir / "checkpoints").glob("checkpoint-*"), key=lambda d: int(d.name.split("-")[-1]))
        final = [run_dir / "adapter"] if (run_dir / "adapter").exists() else []
        if not checkpoints and not final:
            raise FileNotFoundError(f"No checkpoints or adapter in {run_dir}.")
        adapters += checkpoints + final
    if not adapters and not cfg.include_base:
        raise ValueError("Nothing to sample: pass --run_dir or --adapters, or keep --include_base.")
    return adapters


def select_prompts(cfg: SampleConfig, chat_format: ChatFormat) -> list[list[int]]:
    """The first eval chunk of each of the largest channels, cut to whole turns."""
    eval_split = load_from_disk(cfg.dataset_dir)["eval"]
    channel_sizes: dict[str, int] = {}
    first_chunk: dict[str, int] = {}
    for i, channel in enumerate(eval_split["channel"]):
        channel_sizes[channel] = channel_sizes.get(channel, 0) + 1
        first_chunk.setdefault(channel, i)
    channels = sorted(channel_sizes, key=channel_sizes.get, reverse=True)[: cfg.num_prompts]

    prompts = []
    for channel in channels:
        header, chunk_turns = split_chunk(chat_format, eval_split[first_chunk[channel]]["input_ids"])
        ids, turns = list(header), 0
        for turn in chunk_turns:
            if len(ids) + len(turn) > cfg.prompt_tokens and turns > 0:
                break
            ids += turn
            turns += 1
        prompts.append(ids)
    return prompts


def main():
    cfg, sampling = HfArgumentParser((SampleConfig, SamplingConfig)).parse_args_into_dataclasses()
    adapters = collect_adapters(cfg)

    base_names = {model_spec.adapter_base_model(a) for a in adapters}
    if len(base_names) > 1:
        raise RuntimeError(f"Adapters were trained on different base models: {base_names}")
    base_name = base_names.pop() if base_names else model_spec.DEFAULT_MODEL_NAME

    dataset_meta = json.loads((Path(cfg.dataset_dir) / "meta.json").read_text())
    resolve_context_tokens(sampling, dataset_meta["max_length"])
    tokenizer = model_spec.load_tokenizer(base_name)
    chat_format = ChatFormat(tokenizer)
    prompts = select_prompts(cfg, chat_format)

    model, _ = load_for_inference(base_name, cfg.quantization)
    # (label, PEFT adapter name); adapter name None = base model without adapters.
    variants: list[tuple[str, str | None]] = [("base model", None)] if cfg.include_base else []
    for adapter in adapters:
        # PEFT adapter names become module keys, which can't contain dots.
        name = re.sub(r"[^\w-]+", "_", str(adapter)).strip("_")
        if not isinstance(model, PeftModel):
            model = PeftModel.from_pretrained(model, str(adapter), adapter_name=name)
        else:
            model.load_adapter(str(adapter), adapter_name=name)
        variants.append((str(adapter), name))
    model.eval()

    output = (
        Path(cfg.output) if cfg.output else Path(cfg.run_dir or ".") / f"samples_{datetime.now():%Y-%m-%d_%H-%M-%S}.md"
    )
    lines = [f"# Samples ({datetime.now():%Y-%m-%d %H:%M})", ""]

    for p, prompt in enumerate(prompts):
        prompt_conversation = Conversation.from_ids(model, chat_format, prompt, sampling)
        lines += [f"## Prompt {p + 1}", "", "```", prompt_conversation.to_text([prompt]).strip(), "```", ""]
        for label, adapter_name in variants:
            print(f"Prompt {p + 1}/{len(prompts)}: {label}")
            conversation = Conversation.from_ids(model, chat_format, prompt, sampling)
            set_seed(cfg.seed)
            if adapter_name is None and isinstance(model, PeftModel):
                with model.disable_adapter():
                    new_turns = [conversation.generate_turn() for _ in range(cfg.messages)]
            else:
                if adapter_name is not None:
                    model.set_adapter(adapter_name)
                new_turns = [conversation.generate_turn() for _ in range(cfg.messages)]
            lines += [f"### {label}", "", "```", conversation.to_text(new_turns).strip(), "```", ""]

        output.write_text("\n".join(lines))  # write after every prompt, so partial results survive

    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
