import os
import re
import time

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# 初始化 OpenAI 客户端（自动读取环境变量 OPENAI_API_KEY）
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 读取CSV文件
df = pd.read_csv('arxiv_papers.csv')
print(f"共加载{len(df)}篇论文")
#定义评分Prompt

SYSTEM_PROMPT="""你是一名资深的AI领域论文审稿人。请对论文摘要从三个维度进行打分(1.0-10.0)
1.创新性:方法的原创程度
2.逻辑性:论证是否严密
3.实验充分度:实验设计是否合理,对比是否全面

请严格按照以下格式输出,不要有多余内容:
创新性:X
逻辑性:X
实验充分度:X

评分标准(必须严格使用):
1-3分:摘要混乱、方法不清晰、几乎无实验信息
4-5分:思路一般，论证薄弱，实验描述不足
6-7分:合格水平,有一定贡献但不够突出
8分:  较好，方法清晰，实验较完整
9分:  优秀，有明显创新，实验充分
10分: 顶级，接近顶会 oral 水平

注意：
- 大多数 arXiv 预印本应落在 4-8 分
- 只有真正突出的才给 9 分以上
- 三个维度应独立评判，不要给相同分数
"""
def score_abstract(abstract,max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"摘要:{abstract[:1500]}"},  # 截断防止过长
                ],
                temperature=0.5,
                max_tokens=100,
            )
            result = response.choices[0].message.content.strip()
            # 简单解析, 提取分数
            scores = {}
            for line in result.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    num = re.search(r'\d+(?:\.\d+)?', value)
                    if num:
                        scores[key.strip()] = float(num.group())
            return scores
        except Exception as e:
            print(f"第{attempt + 1}次错误: {e}")
            time.sleep(2 ** attempt)#指数回避
    return {"创新性":0,"逻辑性":0,"实验充分度":0}

# 对所有摘要进行评分
print(f"开始评分...预计需要{len(df)*3}秒")
scores_list=[]
test_df=df

for idx,row in tqdm(test_df.iterrows(),total=len(test_df)):
    scores=score_abstract(row['摘要'])
    scores_list.append(scores)
    time.sleep(0.5)
#合并结果
scores_df=pd.DataFrame(scores_list)
result_df=pd.concat([test_df.reset_index(drop=True),scores_df],axis=1)
#保存结果
result_df.to_csv('arxiv_papers_scores.csv',index=False,encoding='utf-8-sig')
print("评分完成,结果已保存到arxiv_papers_scores.csv")

#看一眼统计信息
print("\n统计信息:")
print(result_df[['创新性','逻辑性','实验充分度']].describe())
print("\n平均分:")
print(result_df[['创新性','逻辑性','实验充分度']].mean())
print("\n中位数:")
print(result_df[['创新性','逻辑性','实验充分度']].median())
print("\n最高分:")
print(result_df[['创新性','逻辑性','实验充分度']].max())
print("\n最低分:")
print(result_df[['创新性','逻辑性','实验充分度']].min())
