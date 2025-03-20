from datasets import load_dataset, concatenate_datasets
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
import torch

# Replace dataset loading with the merged file and shuffle it.
dataset = load_dataset("json", data_files={"train": "training_data/merged_train.jsonl"}, split="train")
dataset = dataset.shuffle(seed=42)  # shuffle the dataset

# Prepare tokenizer and model.
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Tokenization function (fine-tune on the combined "prompt" and "generated" text).
def tokenize_function(example):
    text = example["prompt"] + " " + example["generated"]
    tokenized = tokenizer(text, truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()  # add labels for computing loss
    return tokenized

tokenized_dataset = dataset.map(tokenize_function, batched=False)
tokenized_dataset = tokenized_dataset.remove_columns(dataset.column_names)
tokenized_dataset.set_format("torch")

training_args = TrainingArguments(
    output_dir="./finetuned_gpt2",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,  # simulate larger batch size for better learning
    learning_rate=5e-5,  # lowered learning rate for stable training
    save_steps=500,
    save_total_limit=2,
    logging_steps=10,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()

# Save the fine-tuned model and tokenizer.
model.save_pretrained("./finetuned_gpt2")
tokenizer.save_pretrained("./finetuned_gpt2")
