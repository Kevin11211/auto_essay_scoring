import gradio as gr
import torch
from transformers import AutoTokenizer

from models import DebertaV3ForCustomClassification

tokenizer = AutoTokenizer.from_pretrained(
    's986103/DebertaV3ForCustomClassification')

model = DebertaV3ForCustomClassification.from_pretrained(
    's986103/DebertaV3ForCustomClassification')
model.eval()


def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs)
    prediction = torch.argmax(logits, dim=1).item()
    return prediction


iface = gr.Interface(fn=classify_text,
                     inputs="text",
                     outputs="label",
                     description="自動作文評分")

# 啟動 UI
iface.launch()
