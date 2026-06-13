"""Sample a continuing multi-party conversation from the fine-tuned model.

Examples::

    # auto-generate 30 messages in a channel
    uv run python generate.py --prompt "Stockfish - engines-dev" --max-messages 30

    # interactive: you pick the next speaker (blank = let the model choose)
    uv run python generate.py --prompt "Engine Programming - ataxx" --interactive

By default it loads the base model in 4-bit and applies the LoRA adapter given
by --adapter. Point --adapter at a "*_merged" directory (and it will be loaded
directly) or at a "*_LORA" adapter directory.
"""

import argparse
import sys

import torch
from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    TextStreamer,
)
from peft import PeftModel

import constants


def load(adapter_path: str):
    is_merged = adapter_path.rstrip("/").endswith("_merged")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_name = adapter_path if is_merged else constants.model_name
    model = AutoModelForMultimodalLM.from_pretrained(
        base_name,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        device_map="auto",
        token=constants.hf_token,
    )
    if not is_merged:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    processor = AutoProcessor.from_pretrained(adapter_path, token=constants.hf_token)
    return model, processor


def trim_to_window(context: str, header: str, tokenizer, max_tokens: int) -> str:
    """Slide a token window over the context: always keep the system header,
    then drop whole oldest ChatML blocks until the rest fits in max_tokens."""
    if len(tokenizer.encode(context)) <= max_tokens:
        return context

    rest = context[len(header):]  # starts at the first "<|im_start|>" block
    while len(tokenizer.encode(header + rest)) > max_tokens:
        nxt = rest.find(constants.IM_START, len(constants.IM_START))
        if nxt == -1:
            # One block alone is too long: hard-truncate it from the left.
            room = max_tokens - len(tokenizer.encode(header))
            rest = tokenizer.decode(tokenizer.encode(rest)[-room:], skip_special_tokens=False)
            break
        rest = rest[nxt:]
    return header + rest


@torch.no_grad()
def converse(model, processor, channel: str, max_messages: int, interactive: bool):
    tokenizer = processor.tokenizer
    im_end_id = tokenizer.convert_tokens_to_ids(constants.IM_END)
    # Leave room for the tokens we generate each turn.
    window_tokens = constants.max_token_context_length - 512

    header = f"{constants.IM_START}system\nChannel: {channel}{constants.IM_END}\n"
    context = header
    print(context, end="", flush=True)

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)

    for _ in range(max_messages):
        seed = constants.IM_START
        if interactive:
            who = input("\nNext speaker (blank = model picks, 'q' = quit): ").strip()
            if who == "q":
                break
            if who:
                seed += who + "\n"
        context += seed
        print(seed, end="", flush=True)

        # Slide the token window so we never exceed the model's context.
        windowed = trim_to_window(context, header, tokenizer, window_tokens)
        # Text-only: no images passed, so the vision tower is never invoked.
        inputs = processor(text=[windowed], return_tensors="pt").to(model.device)
        output = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            repetition_penalty=1.1,
            max_new_tokens=512,
            eos_token_id=im_end_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            streamer=streamer,
        )

        new_text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        context += new_text
        if not new_text.rstrip().endswith(constants.IM_END):
            context += constants.IM_END
        context += "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="Stockfish - engines-dev",
                        help='Channel header, e.g. "Engine Programming - ataxx"')
    parser.add_argument("--adapter", default=None,
                        help="Path to a *_LORA adapter or *_merged model dir")
    parser.add_argument("--max-messages", type=int, default=30)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    if args.adapter is None:
        print("error: pass --adapter pointing at a trained *_LORA or *_merged dir",
              file=sys.stderr)
        sys.exit(1)

    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name())
    else:
        print("Using CPU")

    model, processor = load(args.adapter)
    converse(model, processor, args.prompt, args.max_messages, args.interactive)


if __name__ == "__main__":
    main()
