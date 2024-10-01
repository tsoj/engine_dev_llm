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
    StoppingCriteria
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
from pathlib import Path

from accelerate import Accelerator
from transformers.generation.configuration_utils import GENERATION_CONFIG_NAME

device_index = Accelerator().process_index
device_map = {"": device_index}

print("device_index:", device_index)

model_name = "Qwen/Qwen2.5-14B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

character_context_length=8192

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
def generate_text(model, tokenizer, prompt, max_new_tokens=1000):

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
            temperature=1.0,
        )
    }

    stops = {
        "normal": ["<|"],
        "user": ["|>", " "],
    }

    current_config = "normal" if prompt.count("|>") >= prompt.count("<|") else "user"

    while True:

        inputs = tokenizer(output, return_tensors="pt").to(model.device)

        #print("Using config:", current_config)
        model_output = model.generate(
            **inputs,
            generation_config=configs[current_config],
            max_length=None,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=[MyStoppingCriteria(stops[current_config], output, tokenizer)],
        )

        output = tokenizer.decode(model_output[0], skip_special_tokens=True)

        #print("Stopped at:", output, "\n-----------------------")

        current_config = "normal" if current_config == "user" else "user"

        num_generated_tokens = len(model_output[0]) - len(tokenizer.encode(prompt))

        print(f"Generated {num_generated_tokens}/{max_new_tokens} tokens")

        if num_generated_tokens >= max_new_tokens:
            break

    return output

# Load the base model for inference
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

# Merge the LoRA weights with the base model
merged_model = PeftModel.from_pretrained(base_model, "./fine_tuned_model_2024-10-1-1822")
merged_model = merged_model.merge_and_unload()

# Create a text generation pipeline
generator = TextGenerationPipeline(model=merged_model, tokenizer=tokenizer)

# Generate text
#prompt = "Stockfish - engines-dev:\n<|tsoj|>\nZuppa, what do you think about leela data?</s>\n\n<|zuppadcipolle|>"
prompt = "Stockfish - engines-dev:\n<|__arandomnoob replies to fuuryy|>\n"

generated_text = generate_text(merged_model, tokenizer, prompt)
print(f"Generated text:\n{generated_text}")

# Generate text using the pipeline (alternative method)
#pipeline_output = generator(prompt, max_length=character_context_length, truncation=True, num_return_sequences=1)
#print(f"Pipeline generated text:\n{pipeline_output[0]['generated_text']}")
