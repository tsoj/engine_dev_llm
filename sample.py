"""Compare checkpoints (and the base model) by continuing held-out conversations.

    ./run.sh sample.py --run_dir runs/my-run

For --num_prompts eval chunks spread evenly over the held-out split, the first
--prompt_tokens tokens are used as the prompt, and every model continues them for
--messages messages with the same random seed. The real continuation from the
dataset is shown alongside. Loss alone says little about whether the output reads
like the real channel; this lets you judge that side by side. The result is
written as a Markdown file.
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
from inference import Conversation, SamplingConfig, load_for_inference, resolve_context_tokens, split_chunk


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
    split: str = field(
        default="eval", metadata={"help": 'Where prompts come from. "train" for datasets without held-out data.'}
    )
    num_prompts: int = 5
    prompt_tokens: int = field(default=512, metadata={"help": "Should leave room for --messages."})
    messages: int = 12
    quantization: model_spec.Quantization | None = None
    seed: int = 0
    output: str | None = field(default=None, metadata={"help": "Defaults to <run_dir>/samples_<time>.md."})


def collect_adapters(run_dir_name: str | None, adapter_paths: list[str], include_base: bool) -> list[Path]:
    """The adapters to compare: the given ones, plus every checkpoint of a run and its final adapter."""
    adapters = [Path(a) for a in adapter_paths]
    if run_dir_name is not None:
        run_dir = Path(run_dir_name)
        checkpoints = sorted((run_dir / "checkpoints").glob("checkpoint-*"), key=lambda d: int(d.name.split("-")[-1]))
        final = [run_dir / "adapter"] if (run_dir / "adapter").exists() else []
        if not checkpoints and not final:
            raise FileNotFoundError(f"No checkpoints or adapter in {run_dir}.")
        adapters += checkpoints + final
    if not adapters and not include_base:
        raise ValueError("Nothing to do: pass --run_dir or --adapters, or keep --include_base.")
    return adapters


def adapter_name(adapter: Path) -> str:
    """PEFT adapter names become module keys, which can't contain dots."""
    return re.sub(r"[^\w-]+", "_", str(adapter)).strip("_")


def select_prompts(cfg: SampleConfig, chat_format: ChatFormat) -> list[tuple[list[int], list[list[int]]]]:
    """Eval chunks spread evenly over the eval split (which is ordered by channel, then time), each cut
    into a prompt of whole turns and the real continuation (up to --messages turns)."""
    dataset = load_from_disk(cfg.dataset_dir)
    if cfg.split not in dataset:
        raise ValueError(f'{cfg.dataset_dir} has no "{cfg.split}" split (it has {list(dataset)}); try --split train.')
    eval_split = dataset[cfg.split]
    n = len(eval_split)
    count = min(cfg.num_prompts, n)
    indices = sorted({round(i * (n - 1) / max(count - 1, 1)) for i in range(count)})

    prompts = []
    for index in indices:
        chunk = eval_split[index]["input_ids"]
        header, chunk_turns = split_chunk(chat_format, chunk)
        # Never use up more than half of a short chunk, so there is a real continuation left to compare against.
        budget = min(cfg.prompt_tokens, len(chunk) // 2)
        ids, used = list(header), 0
        for turn in chunk_turns:
            if len(ids) + len(turn) > budget and used > 0:
                break
            ids += turn
            used += 1
        prompts.append((ids, chunk_turns[used : used + cfg.messages]))
    return prompts


def main():
    cfg, sampling = HfArgumentParser((SampleConfig, SamplingConfig)).parse_args_into_dataclasses()
    adapters = collect_adapters(cfg.run_dir, cfg.adapters, cfg.include_base)

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
        name = adapter_name(adapter)
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

    for p, (prompt, real_turns) in enumerate(prompts):
        prompt_conversation = Conversation.from_ids(model, chat_format, prompt, sampling)
        lines += [f"## Prompt {p + 1}", "", "```", prompt_conversation.to_text([prompt]).strip(), "```", ""]
        lines += ["### Real continuation", "", "```", prompt_conversation.to_text(real_turns).strip(), "```", ""]
        for label, name in variants:
            print(f"Prompt {p + 1}/{len(prompts)}: {label}")
            conversation = Conversation.from_ids(model, chat_format, prompt, sampling)
            set_seed(cfg.seed)
            if name is None and isinstance(model, PeftModel):
                with model.disable_adapter():
                    new_turns = [conversation.generate_turn() for _ in range(cfg.messages)]
            else:
                if name is not None:
                    model.set_adapter(name)
                new_turns = [conversation.generate_turn() for _ in range(cfg.messages)]
            lines += [f"### {label}", "", "```", conversation.to_text(new_turns).strip(), "```", ""]

        output.write_text("\n".join(lines))  # write after every prompt, so partial results survive

    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
