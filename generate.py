"""Simulate a channel with a trained model.

    # let the model write 30 messages
    ./run.sh generate.py --model runs/my-run/adapter

    # pick a channel and take part in the conversation
    ./run.sh generate.py --model runs/my-run/adapter --channel "My Server - general" --interactive

--model can be a LoRA adapter (runs/*/adapter or runs/*/checkpoints/checkpoint-*),
a merged model (runs/*/merged), or a base model name. By default the weights are
loaded in 4-bit, which needs roughly 10 GB of VRAM for Gemma 4 12B.
"""

from dataclasses import dataclass, field

import torch
from transformers import HfArgumentParser, TextStreamer, set_seed

import model_spec
from chat_format import ChatFormat
from inference import Conversation, SamplingConfig, load_for_inference, resolve_context_tokens, trained_dataset


@dataclass
class GenerateConfig:
    model: str = field(metadata={"help": "Adapter dir, merged model dir, or base model name."})
    channel: str | None = field(
        default=None, metadata={"help": "Defaults to the first channel the run was trained on."}
    )
    max_messages: int = 30
    interactive: bool = False
    quantization: model_spec.Quantization | None = field(
        default=None, metadata={"help": 'Defaults to "4bit" on GPU and "none" on CPU.'}
    )
    seed: int | None = None


def main():
    cfg, sampling = HfArgumentParser((GenerateConfig, SamplingConfig)).parse_args_into_dataclasses()
    if cfg.seed is not None:
        set_seed(cfg.seed)

    print("Using GPU:", torch.cuda.get_device_name() if torch.cuda.is_available() else "none (CPU)")
    dataset_meta = trained_dataset(cfg.model)
    if dataset_meta is None and sampling.context_tokens is None:
        sampling.context_tokens = 1024
        print("Not a runs/ directory, using --context_tokens 1024.")
    resolve_context_tokens(sampling, dataset_meta["max_length"] if dataset_meta else None)

    channels = (dataset_meta or {}).get("channels", [])
    channel = cfg.channel or next(iter(channels), None)
    if channel is None:
        raise ValueError("Pass --channel: the run directory doesn't say which channels it was trained on.")
    if channels and channel not in channels:
        print(f"Warning: {channel!r} is not one of the {len(channels)} trained channels, e.g. {channels[:3]}.")
    elif cfg.channel is None and channels:
        print(f"Using the first of {len(channels)} trained channels; pass --channel to pick another.")

    model, tokenizer = load_for_inference(cfg.model, cfg.quantization)
    chat_format = ChatFormat(tokenizer)
    conversation = Conversation(model, chat_format, chat_format.header_ids(channel), sampling)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    print(f"\n# {channel}")
    if cfg.interactive:
        print(
            "Enter: blank = model picks the next speaker, 'name' = model writes as name, "
            "'name: message' = you write as name, 'q' = quit."
        )

    for _ in range(cfg.max_messages):
        role = ""
        if cfg.interactive:
            line = input("\n> ").strip()
            if line == "q":
                break
            if ": " in line:
                author, content = line.split(": ", 1)
                conversation.add_message(author, content)
                continue
            role = line
        print(f"\n{role}" if role else "", flush=True)
        conversation.generate_turn(role, streamer=streamer)


if __name__ == "__main__":
    main()
