import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset,DatasetDict
from transformers import AutoTokenizer

#加载数据
df=pd.read_csv('arxiv_papers_scores.csv')
print(f"共加载了{len(df)}篇数据")
#检查是否有空值
df=df.dropna(subset=['摘要','创新性','逻辑性','实验充分度'])
print(f"删除空值后，共保留了{len(df)}篇数据")

#此处做的是回归任务（预测论文分数），不需要分类标签
#三个分数转换为三个目标值
features=df["摘要"].tolist()
labels=df[["创新性","逻辑性","实验充分度"]].values.tolist()

#划分训练集和测试集
X_train,X_val,y_train,y_val=train_test_split(
    features,labels,test_size=0.2,random_state=42
)
#包装成Hugging Face的Dataset对象
train_dataset=Dataset.from_dict({
    "abstract":X_train,
    "labels":y_train
})
val_dataset=Dataset.from_dict({
    "abstract":X_val,
    "labels":y_val
})
#创建DatasetDict对象
dataset=DatasetDict({
    "train":train_dataset,
    "val":val_dataset
})
#加载tokenizer
tokenizer=AutoTokenizer.from_pretrained("bert-base-chinese")

def tokenize_function(examples):
    return tokenizer(
        examples["abstract"],
        padding="max_length",
        truncation=True,
        max_length=512
    )
#对数据集做分词
tokenized_dataset=dataset.map(tokenize_function,batched=True)
tokenized_dataset=tokenized_dataset.remove_columns(['abstract'])
tokenized_dataset.set_format('torch')
#保留预处理后的数据，方便下次直接加载

tokenized_dataset.save_to_disk("tokenized_data")
print("预处理完成,已保存到tokenized_data/")

print(f"训练集样本数:{len(tokenized_dataset['train'])}")
print(f"验证集样本数:{len(tokenized_dataset['val'])}")
print(f"输入维度:{tokenized_dataset['train'][0]['input_ids'].shape}")
print(f"标签示例:{tokenized_dataset['train'][0]['labels']}")
