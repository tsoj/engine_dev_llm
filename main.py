import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

model_name = "distilbert/distilgpt2"#"meta-llama/Meta-Llama-3-8B"#"Qwen/Qwen2.5-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

bnb_config = BitsAndBytesConfig(
    # load_in_4bit=True,
    # bnb_4bit_quant_type="nf4",
    # bnb_4bit_use_double_quant=True,
    # bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=16,
    lora_alpha=8,
    # target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)


dataset = load_dataset("text", data_files={"train": "./lots_of_text.txt"})
print(dataset)
