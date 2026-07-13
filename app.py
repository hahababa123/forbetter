# import gradio as gr
import streamlit as st
from transformers import AutoTokenizer,AutoModelForSequenceClassification
import torch

model_path="./bert_scorer_model"
tokenizer=AutoTokenizer.from_pretrained(model_path)
model=AutoModelForSequenceClassification.from_pretrained(model_path)

def score(text):
    inputs=tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    )
    with torch.no_grad():
        scores=model(**inputs).logits.squeeze().tolist()
    return {
        "创新性":round(scores[0],1),
        "逻辑性":round(scores[1],1),
        "实验充分度":round(scores[2],1)
    }

st.title("AI论文评分器")
text=st.text_area("输入论文摘要")
if st.button("评分"):
    result=score(text)
    st.json(result)

# demo=gr.Interface(
#     fn=score,
#     inputs=gr.Textbox(label="输入论文摘要",lines=5,placeholder="粘贴你的论文摘要..."),
#     outputs=gr.JSON(label="评分结果"),
#     title="AI论文评分器",
#     description="基于BERT微调,从创新性,逻辑性,实验充分度三个维度自动评分"
# )
# demo.launch(share=True)