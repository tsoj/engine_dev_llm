import math
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TextGenerationPipeline
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
from pathlib import Path

from accelerate import Accelerator

device_index = Accelerator().process_index
device_map = {"": device_index}

print("device_index:", device_index)

model_name = "Qwen/Qwen2.5-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

character_context_length=8192

# Text generation setup
def generate_text(model, tokenizer, prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Load the base model for inference
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

# Merge the LoRA weights with the base model
merged_model = PeftModel.from_pretrained(base_model, "./fine_tuned_model")
merged_model = merged_model.merge_and_unload()

# Create a text generation pipeline
generator = TextGenerationPipeline(model=merged_model, tokenizer=tokenizer)

# Generate text
prompt = "Stockfish - engines-dev:\n<|zuppadcipolle|>"

# generated_text = generate_text(merged_model, tokenizer, prompt)
# print(f"Generated text:\n{generated_text}")

# Generate text using the pipeline (alternative method)
pipeline_output = generator(prompt, max_length=character_context_length, truncation=True, num_return_sequences=1)
print(f"Pipeline generated text:\n{pipeline_output[0]['generated_text']}")
