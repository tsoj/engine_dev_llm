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

# model_name = "distilbert/distilgpt2"
model_name = "Qwen/Qwen2.5-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

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
def load_and_chunk_dataset(file_path, chunk_size, overlap, test_size):
    with open(file_path, 'r') as file:
        text = file.read()

    # Create chunks of text
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)

    # Determine split point
    split_point = math.floor(len(chunks) * (1 - test_size))

    # Create train and test datasets
    train_dataset = Dataset.from_dict({"text": chunks[:split_point]})
    test_dataset = Dataset.from_dict({"text": chunks[split_point:]})

    return {"train": train_dataset, "test": test_dataset}

# Load and preprocess your dataset
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=2048)

# Load and split the dataset
dataset = load_and_chunk_dataset("lots_of_text.txt", chunk_size=1000, overlap=100, test_size=0.1)
# Tokenize the datasets
tokenized_dataset = {
    "train": dataset["train"].map(preprocess_function, batched=True, remove_columns=["text"]),
    "test": dataset["test"].map(preprocess_function, batched=True, remove_columns=["text"])
}

# Set up the trainer
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,  # Add this line
    gradient_accumulation_steps=8,
    warmup_steps=100,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    eval_strategy="steps",  # Add this line
    eval_steps=20,  # Add this line
    load_best_model_at_end=True,  # Add this line
    metric_for_best_model="eval_loss",  # Add this line
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],  # Add this line
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

# # Create a text generation pipeline
# generator = TextGenerationPipeline(model=merged_model, tokenizer=tokenizer)  # Adjust device as needed

# # Generate text
# prompt = "Requirement already satisfied"
# generated_text = generate_text(merged_model, tokenizer, prompt)
# print(f"Generated text:\n{generated_text}")

# Generate text using the pipeline (alternative method)
pipeline_output = generator(prompt, max_length=100, truncation=True, num_return_sequences=1)
print(f"Pipeline generated text:\n{pipeline_output[0]['generated_text']}")
