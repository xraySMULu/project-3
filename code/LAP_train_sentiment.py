# Force transformers to ignore TensorFlow
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2ForSequenceClassification,
    TrainingArguments,
    Trainer
)
import evaluate
import torch

# 1. Load Dataset
print("🔄 Loading dataset...")
dataset = load_dataset("mteb/tweet_sentiment_extraction")

# ✅ Slice early to shrink dataset for fast training
dataset["train"] = dataset["train"].select(range(500))
dataset["test"] = dataset["train"].select(range(100))  # Reuse small slice for eval

train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# 2. Load Tokenizer
print("🔄 Loading GPT-2 tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# 3. Remove empty or invalid text entries
def is_not_empty(example):
    return example["text"] is not None and example["text"].strip() != ""

train_dataset = train_dataset.filter(is_not_empty)
eval_dataset = eval_dataset.filter(is_not_empty)

# 4. Tokenize
def tokenize(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=64  # ✅ Reduced for faster training
    )

train_dataset = train_dataset.map(tokenize)
eval_dataset = eval_dataset.map(tokenize)

# 5. Remove bad tokenized samples
def has_valid_input(example):
    return (
        "input_ids" in example and len(example["input_ids"]) == 64 and
        "attention_mask" in example and len(example["attention_mask"]) == 64
    )

train_dataset = train_dataset.filter(has_valid_input)
eval_dataset = eval_dataset.filter(has_valid_input)

# 6. Remove invalid or out-of-range labels
def has_valid_label(example):
    return "label" in example and isinstance(example["label"], int) and 0 <= example["label"] <= 2

train_dataset = train_dataset.filter(has_valid_label)
eval_dataset = eval_dataset.filter(has_valid_label)

# 7. Final tensor validation
def remove_zero_length_tensors(dataset, name="train"):
    valid_indexes = []
    for i, sample in enumerate(dataset):
        try:
            if (
                isinstance(sample["input_ids"], torch.Tensor)
                and isinstance(sample["attention_mask"], torch.Tensor)
                and sample["input_ids"].shape[0] == 64
                and sample["attention_mask"].shape[0] == 64
                and isinstance(sample["label"], torch.Tensor)
            ):
                valid_indexes.append(i)
            else:
                print(f"🚫 Removed malformed sample at index {i} in {name} set.")
        except Exception as e:
            print(f"❌ Exception at index {i} in {name} set: {e}")
    print(f"✅ {name} set cleaned: {len(valid_indexes)} samples kept.")
    return dataset.select(valid_indexes)

# 8. Set format and strip unwanted columns
columns_to_keep = {"input_ids", "attention_mask", "label"}
train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if col not in columns_to_keep])
eval_dataset = eval_dataset.remove_columns([col for col in eval_dataset.column_names if col not in columns_to_keep])

train_dataset.set_format("torch")
eval_dataset.set_format("torch")

train_dataset = remove_zero_length_tensors(train_dataset, "train")
eval_dataset = remove_zero_length_tensors(eval_dataset, "eval")

# ✅ Confirm final dataset sizes
print(f"📊 Final train dataset size: {len(train_dataset)}")
print(f"📊 Final eval dataset size: {len(eval_dataset)}")

# 9. Load Model
print("🔄 Initializing GPT-2 model for sequence classification...")
model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=3)
model.config.pad_token_id = model.config.eos_token_id

# 10. Evaluation Metric
metric = evaluate.load("accuracy")

def compute_metrics(pred):
    logits, labels = pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 11. Training Configuration
training_args = TrainingArguments(
    output_dir="./models/gpt2_sentiment",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,  # ✅ Smaller batch for CPU
    per_device_eval_batch_size=4,
    num_train_epochs=1,  # ✅ One quick epoch
    weight_decay=0.01,
    save_total_limit=1,
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=20,
    load_best_model_at_end=True,
    report_to="none"
)

# 12. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# 13. Train
print("🚀 Starting quick training run...")
trainer.train()

# 14. Save Model
print("✅ Training complete. Saving model...")
model.save_pretrained("./models/gpt2_sentiment")
tokenizer.save_pretrained("./models/gpt2_sentiment")
