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
        self.stops = stops,
        self.tokenizer = tokenizer
        self.stop_counter = {}
        for stop in self.stops:
            self.stop_counter[stop] = text_so_far.count(stop)

    def __call__(self, input_ids, scores, **kwargs):
        generated_text = self.tokenizer.decode(input_ids[0])
        for stop in self.stops:
            if self.stop_counter[stop] < generated_text.count(stop):
                return False
        return True

# Text generation setup
@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_length=1000):

    output = prompt

    configs = {
        "normal": GenerationConfig(
            do_sample=False,
            dola_layers="low",
            repetition_penalty=2.0
        ),
        "user": GenerationConfig(
            do_sample=True,
            num_beams=1,
            temperature=2.0,
        )
    }

    stops = {
        "normal": ["<|"],
        "user": ["|>", " "],
    }

    current_config = "normal"

    while len(output) < max_length:

        inputs = tokenizer(output, return_tensors="pt").to(model.device)

        print("Using config:", current_config)
        model_output = model.generate(
            **inputs,
            generation_config=configs[current_config],
            max_length=max_length,
            max_new_tokens=None,
            stopping_criteria=[MyStoppingCriteria(stops[current_config], output, tokenizer)],
            #num_return_sequences=1,
            #temperature=0.7,
            #top_k=50,
            #top_p=0.95,
            # do_sample=False,
            # dola_layers="low",
            # repetition_penalty=2.0,
            #penalty_alpha=0.6, top_k=4,
        )

        output = tokenizer.decode(model_output[0], skip_special_tokens=True)

        print("Stopped at:", output)

        current_config = "normal" if current_config == "user" else "user"

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
prompt = "Stockfish - engines-dev:\n<|ciekce|>\n"

generated_text = generate_text(merged_model, tokenizer, prompt)
print(f"Generated text:\n{generated_text}")

# Generate text using the pipeline (alternative method)
#pipeline_output = generator(prompt, max_length=character_context_length, truncation=True, num_return_sequences=1)
#print(f"Pipeline generated text:\n{pipeline_output[0]['generated_text']}")
