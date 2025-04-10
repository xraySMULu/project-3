# sentiment_model.py

import torch
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification

model_path = "../models/gpt2_sentiment_quantized"  # ✅ Smaller quantized model
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2ForSequenceClassification.from_pretrained(model_path)
model.eval()

label_map = {0: "negative", 1: "neutral", 2: "positive"}

def predict_sentiment(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=64
    )
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()
    return label_map[prediction]

