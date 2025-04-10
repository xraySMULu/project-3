import torch
from transformers import GPT2ForSequenceClassification

model = GPT2ForSequenceClassification.from_pretrained("./models/gpt2_sentiment")
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
quantized_model.save_pretrained("./models/gpt2_sentiment_quantized")
