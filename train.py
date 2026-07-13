import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import load_from_disk
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
#加载预处理好的数据
tokenized_dataset=load_from_disk("tokenized_data")
#加载模型
model=AutoModelForSequenceClassification.from_pretrained(
    "bert-base-chinese",
    num_labels=3
)
tokenizer=AutoTokenizer.from_pretrained("bert-base-chinese")
#定义评分指标
def compute_metrics(eval_pred):
    predictions,labels=eval_pred
    #predictions是logits,直接取数值
    mse=mean_squared_error(labels,predictions)
    mae=mean_absolute_error(labels,predictions)
    r2=r2_score(labels,predictions)
    return{
        "mse":mse,
        "mae":mae,
        "r2":r2
    }
#训练参数配置
training_args=TrainingArguments(
    output_dir="./bert_scorer_results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="mse",
    greater_is_better=False,
    save_total_limit=2,
    report_to="none"
)
trainer=Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["val"],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)
#开始训练
print("开始训练BERT回归模型")
trainer.train()

#保存最终模型
model.save_pretrained("./bert_scorer_model")
tokenizer.save_pretrained("./bert_scorer_model")
print("训练完成,模型已保存到./bert_scorer_model")