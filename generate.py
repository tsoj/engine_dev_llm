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
from pathlib import Path
import sys
import constants
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel

if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name())
else:
    print("Using CPU")


def truncate_string(input_string, max_length):
    lines = input_string.split('\n')
    first_line = lines[0] + '\n\n'
    remaining_text = '\n'.join(lines[1:])

    while len(first_line) + len(remaining_text) > max_length or not remaining_text.startswith('<|'):
        if len(remaining_text) > 0:
            remaining_text = remaining_text[1:]
        else:
            break

    return first_line + remaining_text

def count_chars_before_last_pipe_greater(input_string):
    last_pipe_greater_index = input_string.rfind('|>')
    if last_pipe_greater_index == -1:
        return 0  # No '|>' found in the string
    return len(input_string) - last_pipe_greater_index - 2

def is_inside_message(input_string):
    return input_string.rfind('<|') <= input_string.rfind('|>')

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
        if is_inside_message(generated_text) and count_chars_before_last_pipe_greater(generated_text) >= constants.max_single_message:
            return True
        return False

# Text generation setup
@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_new_chars, interactive=False):

    num_generated_chars = 0

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

        if interactive and not skipping and len(output) >= 2 and output[-2:] == "<|":
            if current_config != "user":
                print(f"\n\nWARNING: Entering username mode even though config is set to \"{current_config}\"\n\n")

            sys.stdout.write("\r\033[K")
            sys.stdout.write("Next user ('a' for auto mode, '' to skip, 'q' to quit): ")
            sys.stdout.flush()

            # Get user input
            user_input = input()

            # Erase the line
            sys.stdout.write("\033[1A")  # Move cursor up one line
            sys.stdout.write("\033[K")   # Erase the line
            sys.stdout.flush()

            if user_input == "q":
                break
            if user_input in ["a", ""]:
                sys.stdout.write("<|")
                sys.stdout.flush()
                if user_input == "a":
                    interactive = False
                if user_input == "":
                    skipping = True
                continue

            new_output = user_input.split()[0]
            sys.stdout.write("<|" + new_output)
            sys.stdout.flush()
            output += new_output
        else:
            output = truncate_string(output, constants.max_character_context_length)

            inputs = tokenizer(output, return_tensors="pt").to(model.device)

            streamer = TextStreamerWithNoNewline(tokenizer, skip_prompt=not first_iteration)
            first_iteration = False


            #print("Using config:", current_config)
            model_output = model.generate(
                **inputs,
                streamer=streamer,
                generation_config=configs[current_config],
                max_length=None,
                max_new_tokens=constants.max_character_context_length // 2,
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=[MyStoppingCriteria(stops[current_config], output, tokenizer)],
            )

            previous_output = output
            output = tokenizer.decode(model_output[0], skip_special_tokens=True)

            num_generated_chars += len(output) - len(previous_output)

            if is_inside_message(output) and count_chars_before_last_pipe_greater(output) >= constants.max_single_message:
                output += "\n\n<|"
                sys.stdout.write("\n\n<|")
                sys.stdout.flush()


            if not interactive and num_generated_chars >= max_new_chars:
                break

        current_config = "normal" if current_config == "user" else "user"
        skipping = False


    print("\n----------------------")

model_name = "engine_dev_model_2025-02-28-23-04-06_LORA"


# Define the quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load the base model with quantization
base_model = AutoModelForCausalLM.from_pretrained(
    constants.model_name,
    quantization_config=bnb_config,
    device_map='auto',
    trust_remote_code=True  # If required for some models
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(constants.model_name)

# Apply the LoRA adapters
model = PeftModel.from_pretrained(base_model, "./" + model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Generate text
prompt = "Stockfish - engines-dev"
interactive = "interactive" in sys.argv[1:]
max_new_chars = 1_000_000

for arg in sys.argv[1:]:
    if arg.isdigit():
        max_new_chars = int(arg)
    elif arg != "interactive":
        prompt = arg

print("interactive:", interactive)
print("max_new_tokens:", max_new_chars)
print(f"prompt: \"{prompt}\"")


generate_text(model, tokenizer, prompt, max_new_chars=max_new_chars, interactive=interactive)
