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
from datetime import datetime
import constants
import os

# Find latest checkpoint if it exists
def get_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    if not checkpoint_dir.exists():
        return None

    checkpoints = [d for d in checkpoint_dir.iterdir() if d.name.startswith("checkpoint-")]
    if not checkpoints:
        return None

    # Sort checkpoints by number
    return max(checkpoints, key=lambda x: int(x.name.split("-")[1]))

out_model_name = "engine_dev_model_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
print("out_model_name:", out_model_name)

# Define checkpoint directory
checkpoint_dir = Path("./checkpoints")
checkpoint_dir.mkdir(exist_ok=True)

latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
    print(f"IMPORTANT: Found existing checkpoint at {latest_checkpoint}. Will resume training...")
else:
    print("IMPORTANT: Starting training from scratch ...")

tokenizer = AutoTokenizer.from_pretrained(constants.model_name, token=constants.token)
tokenizer.pad_token = tokenizer.eos_token


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    constants.model_name,
    quantization_config=bnb_config,
    device_map="auto",
    token=constants.token,
)

model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
model.gradient_checkpointing_disable()

# Load and preprocess your dataset
def load_and_chunk_dataset(data_path, char_chunk_size, char_overlap, test_train_ratio, tokenizer):

    train_chunks = []
    test_chunks = []
    for file_path in Path(data_path).glob("*-dev.txt"):
        print("Loading from", file_path)

        with open(file_path, 'r') as file:
            text = file.read()

            split_point = math.floor(len(text) * (1 - test_train_ratio))
            header_text = Path(file_path).stem + ":\n"
            current_chunk_size = char_chunk_size - len(header_text)

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
                assert len(tokenizer.encode(chunk)) <= tokenizer.model_max_length
                if pos < split_point:
                    train_chunks.append(chunk)
                else:
                    test_chunks.append(chunk)
                pos = chunk_end - char_overlap
                assert pos > 0
                if pos <= chunk_start:
                    break

    # Create train and test datasets
    train_dataset = Dataset.from_dict({"text": train_chunks})
    test_dataset = Dataset.from_dict({"text": test_chunks})

    return {"train": train_dataset, "test": test_dataset}

# Load and preprocess your dataset
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=constants.character_context_length)

# Load and split the dataset
dataset = load_and_chunk_dataset(
    "data/text",
    char_chunk_size=constants.character_context_length,
    char_overlap=constants.character_context_length//48,
    test_train_ratio=0.01,
    tokenizer=tokenizer
)

# Tokenize the datasets
tokenized_dataset = {
    "train": dataset["train"].map(preprocess_function, batched=True, remove_columns=["text"]),
    "test": dataset["test"].map(preprocess_function, batched=True, remove_columns=["text"])
}

# Set up the trainer
training_args = TrainingArguments(
    output_dir=str(checkpoint_dir),
    num_train_epochs=2,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    gradient_checkpointing=False,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    eval_accumulation_steps=50,
    warmup_ratio=0.1,
    weight_decay=0.01,
    learning_rate=5e-5,
    lr_scheduler_type="linear",
    fp16=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=400,
    save_strategy="steps",
    save_steps=400,
    save_total_limit=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# Start training
trainer.train(resume_from_checkpoint=str(latest_checkpoint) if latest_checkpoint else None )

test_results = trainer.evaluate()
print(f"Final test loss: {test_results['eval_loss']}")

merged_model = model.merge_and_unload()

merged_model.save_pretrained("./" + out_model_name)
tokenizer.save_pretrained("./" + out_model_name)

print("Finished :D")
