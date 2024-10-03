import math
import torch
from datasets import Dataset
from torch.cuda import temperature
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TextGenerationPipeline,
    GenerationConfig,
    StoppingCriteria,
    TextStreamer
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
from pathlib import Path
import sys
from accelerate import Accelerator
from transformers.generation.configuration_utils import GENERATION_CONFIG_NAME

device_index = Accelerator().process_index
device_map = {"": device_index}

print("device_index:", device_index)

class TextStreamerWithNoNewline(TextStreamer):
    def __init__(self, tokenizer, skip_prompt: bool):
        super().__init__(tokenizer=tokenizer, skip_prompt=skip_prompt)

    def on_finalized_text(self, text: str, stream_end: bool = False):
        stream_end = False
        print(text, flush=True, end="" if not stream_end else None)


class MyStoppingCriteria(StoppingCriteria):
    def __init__(self, stops, text_so_far, tokenizer):
        self.stops = stops
        self.tokenizer = tokenizer
        self.stop_counter = {}
        for stop in self.stops:
            self.stop_counter[stop] = text_so_far.count(stop)

    def __call__(self, input_ids, scores, **kwargs):
        generated_text = self.tokenizer.decode(input_ids[0])
        for stop in self.stops:
            if self.stop_counter[stop] < generated_text.count(stop):
                return True
        return False

# Text generation setup
@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_new_tokens=1000, interactive=False):

    output = prompt

    configs = {
        "normal": GenerationConfig(
            do_sample=False,
            dola_layers="low",
            repetition_penalty=2.0,
        ),
        "user": GenerationConfig(
            do_sample=True,
            num_beams=1,
            temperature=1.2,
        )
    }

    stops = {
        "normal": ["<|"],
        "user": ["|>", " "],
    }

    current_config = "normal" if prompt.count("|>") >= prompt.count("<|") else "user"

    print("----------------------")
    first_iteration = True
    skipping = False
    while True:

        if interactive and current_config == "user" and not skipping:
            assert len(output) >= 3 and output[-3:] == "\n<|", output

            sys.stdout.write("\r\033[K")
            sys.stdout.write("Next user ('a' for auto mode, 's' to skip, 'q' to quit): ")
            sys.stdout.flush()

            # Get user input
            user_input = input()

            # Erase the line
            sys.stdout.write("\033[1A")  # Move cursor up one line
            sys.stdout.write("\033[K")   # Erase the line
            sys.stdout.flush()

            if user_input == "q":
                break
            if user_input in ["a", "s"]:
                sys.stdout.write("<|")
                sys.stdout.flush()
                if user_input == "a":
                    interactive = False
                if user_input == "s":
                    skipping = True
                continue

            new_output = user_input.split()[0]
            sys.stdout.write("<|" + new_output)
            sys.stdout.flush()
            output += new_output
        else:
            inputs = tokenizer(output, return_tensors="pt").to(model.device)

            streamer = TextStreamerWithNoNewline(tokenizer, skip_prompt=not first_iteration)
            first_iteration = False


            #print("Using config:", current_config)
            model_output = model.generate(
                **inputs,
                streamer=streamer,
                generation_config=configs[current_config],
                max_length=None,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=[MyStoppingCriteria(stops[current_config], output, tokenizer)],
            )

            previous_output = output
            output = tokenizer.decode(model_output[0], skip_special_tokens=True)

            #print("Stopped at:", output, "\n-----------------------")

            num_new_generated_tokens = len(model_output[0]) - len(tokenizer.encode(previous_output))
            num_total_generated_tokens = len(model_output[0]) - len(tokenizer.encode(prompt))

            if (num_new_generated_tokens if interactive else num_total_generated_tokens) >= max_new_tokens:
                break

        current_config = "normal" if current_config == "user" else "user"
        skipping = False


    print("\n----------------------")

    return output

model_name = "./engine_dev_model_2024-10-02-13-58-24"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load the base model for inference
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)


#model.dequantize()



# Generate text
prompt = "Stockfish - engines-dev:\n"

interactive = "interactive" in sys.argv[1:]
max_new_tokens = 1000
for arg in sys.argv[1:]:
    if arg.isdigit():
        max_new_tokens = int(arg)
    elif arg != "interactive":
        prompt = arg

print("interactive:", interactive)
print("max_new_tokens:", max_new_tokens)
print(f"prompt: \"{prompt}\"")


generate_text(model, tokenizer, prompt, max_new_tokens=max_new_tokens, interactive=interactive)
