import arxiv
import pandas as pd
from datetime import datetime
#搜索配置
search = arxiv.Search(
    query = "cat:cs.LG OR cat:cs.AI",  # 机器学习/人工智能
    max_results = 200,
    sort_by = arxiv.SortCriterion.SubmittedDate
)

# 下载数据
papers = []
for paper in arxiv.Client(delay_seconds=3.0).results(search):
    papers.append({
        "标题": paper.title,
        "摘要": paper.summary,
        "作者": ", ".join(a.name for a in paper.authors),
        "提交时间": paper.published.strftime("%Y-%m-%d"),
        "论文链接": paper.entry_id
    })
    print(f"已下载: {paper.title[:30]}...")  # 进度提示

# 保存为CSV（后续评分直接读这个文件）
df = pd.DataFrame(papers)
df.to_csv("arxiv_papers.csv", index=False, encoding="utf-8-sig")
print(f"\n✅ 完成！共下载 {len(df)} 篇，已保存到 arxiv_papers.csv")