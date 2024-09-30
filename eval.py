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

character_context_length=6144#4096#8192

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    #quantization_config=bnb_config,
    #device_map="auto",
    device_map=device_map,
)

#model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=16,
    lora_alpha=8,
    # target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

#model = get_peft_model(model, config)
model.gradient_checkpointing_disable()

# Load and preprocess your dataset
def load_and_chunk_dataset(data_path, chunk_size, overlap, test_size):

    train_chunks = []
    test_chunks = []
    for file_path in Path(data_path).glob("*.txt"):
        print("Loading from", file_path)

        with open(file_path, 'r') as file:
            text = file.read()

            split_point = math.floor(len(text) * (1 - test_size))
            header_text = Path(file_path).stem + ":\n"
            current_chunk_size = chunk_size - len(header_text)

            def find_next_start(pos):
                    next_start = text.find("<|", pos)
                    return next_start if next_start != -1 else len(text)

            pos = 0
            while pos < len(text):
                chunk_start = find_next_start(pos)
                chunk_end = min(len(text), chunk_start + current_chunk_size)
                if chunk_start >= len(text):
                    break
                assert chunk_start < chunk_end <= len(text)
                chunk = header_text + text[chunk_start:chunk_end]
                if pos < split_point:
                    train_chunks.append(chunk)
                else:
                    test_chunks.append(chunk)
                pos = chunk_end - overlap
                assert pos > 0
                if pos <= chunk_start:
                    break

    # Create train and test datasets
    train_dataset = Dataset.from_dict({"text": train_chunks})
    test_dataset = Dataset.from_dict({"text": test_chunks})

    return {"train": train_dataset, "test": test_dataset}

# Load and preprocess your dataset
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=character_context_length)

# Load and split the dataset
dataset = load_and_chunk_dataset("data/text", chunk_size=character_context_length, overlap=character_context_length//8, test_size=0.05)
# Tokenize the datasets
tokenized_dataset = {
    "train": dataset["train"].map(preprocess_function, batched=True, remove_columns=["text"]),
    "test": dataset["test"].map(preprocess_function, batched=True, remove_columns=["text"])
}

# Set up the trainer
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=False,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    eval_accumulation_steps=50,
    warmup_steps=500,
    learning_rate=8e-5,
    fp16=True,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    lr_scheduler_type="linear",
)

merged_model = PeftModel.from_pretrained(model, "./fine_tuned_model")
merged_model = merged_model.merge_and_unload()


trainer = Trainer(
    model=merged_model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

test_results = trainer.evaluate()
print(f"Final test loss: {test_results['eval_loss']}")
