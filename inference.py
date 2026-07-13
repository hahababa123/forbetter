import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification

model_path="./bert_scorer_model"
tokenizer=AutoTokenizer.from_pretrained(model_path)
model=AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

def score_abstract(text):
    inputs=tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    )
    with torch.no_grad():
        outputs=model(**inputs)
        scores=outputs.logits.squeeze().tolist()
    return {
        "创新性":round(scores[0],1),
        "逻辑性":round(scores[1],1),
        "实验充分度":round(scores[2],1)
    }

test="We propose a new method for image classification using attention mechanism"
print(score_abstract(test))
