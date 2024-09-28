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

# model_name = "distilbert/distilgpt2"
model_name = "Qwen/Qwen2.5-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

character_context_length=4096

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
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
dataset = load_and_chunk_dataset("data/text", chunk_size=character_context_length, overlap=character_context_length//8, test_size=0.1)
# Tokenize the datasets
tokenized_dataset = {
    "train": dataset["train"].map(preprocess_function, batched=True, remove_columns=["text"]),
    "test": dataset["test"].map(preprocess_function, batched=True, remove_columns=["text"])
}

# Set up the trainer
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    warmup_steps=500,
    learning_rate=8e-5,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    lr_scheduler_type="linear",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# Start training
trainer.train()

test_results = trainer.evaluate()
print(f"Final test loss: {test_results['eval_loss']}")

model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")


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
    quantization_config=bnb_config,
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
prompt = "Requirement already satisfied"

# generated_text = generate_text(merged_model, tokenizer, prompt)
# print(f"Generated text:\n{generated_text}")

# Generate text using the pipeline (alternative method)
pipeline_output = generator(prompt, max_length=character_context_length, truncation=True, num_return_sequences=1)
print(f"Pipeline generated text:\n{pipeline_output[0]['generated_text']}")
