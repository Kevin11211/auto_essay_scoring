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
                       truncation=True, max_length=1024)
    with torch.no_grad():
        logits = model(**inputs)
    prediction = torch.argmax(logits, dim=-1).item() + 1
    return prediction


# 自定义 CSS 样式
custom_css = """
#input_textbox textarea {
    border: 2px solid #1E90FF !important; /* 设置输入框边框颜色为蓝色 */
    border-radius: 10px !important;       /* 设置边框圆角 */
}

#output_textbox textarea {
    border: 2px solid #FFA500 !important; /* 设置输出框边框颜色为橘色 */
    border-radius: 10px !important;       /* 设置边框圆角 */
    font-size: 24px !important;           /* 设置字体大小为24px */
    text-align: center !important;        /* 将文本居中对齐 */
    display: flex;
    justify-content: center;
    align-items: center;
}
"""

# 定义 Gradio 接口
iface = gr.Interface(
    fn=classify_text,
    inputs=gr.Textbox(label="請輸入文章", elem_id="input_textbox"),
    outputs=gr.Textbox(label="評分結果(1-6)", elem_id="output_textbox"),
    description="自動作文評分",
    css=custom_css  # 将自定义 CSS 添加到 Gradio 应用
)

# 啟動 UI
iface.launch()
