# 检索结果评测

该目录只评测知识库检索，不调用最终答案生成，不评价路由或Agent端到端效果。

## 评测内容

- 30条人工标注问题，覆盖当前12份企业知识文档。
- Baseline：FAISS直接返回Top 5。
- Current：FAISS召回候选后，使用当前配置的Reranker返回Top 5。
- 指标：Precision@5、Recall@5、HitRate@5、MRR@5、NDCG@5、平均耗时和P95耗时。
- 索引检查：统计索引来源、标注Chunk缺失情况和疑似乱码Chunk。

相关性标签使用当前冻结索引中的 doc_id + chunk_index。如果重新上传文档、重建索引或改变分块参数，需要重新核对标注。

## 运行

在 python-service 目录执行：

~~~powershell
python -m evaluation.run_retrieval_eval
~~~

调试前5条：

~~~powershell
python -m evaluation.run_retrieval_eval --limit 5
~~~

指定输出目录或Top K：

~~~powershell
python -m evaluation.run_retrieval_eval --top-k 5 --output-dir evaluation/results/manual
~~~

结果目录包含：

- results.json：逐题召回结果、指标、耗时、错误和索引健康信息。
- report.md：适合人工查看及面试展示的汇总报告。

仓库保留的代表性结果见 [benchmark.md](benchmark.md)。`results/` 是每次运行自动生成的产物，默认不提交到 Git。

## 指标解释

- Precision@5：返回的5个Chunk中，人工标注为相关的比例。
- Recall@5：所有人工标注相关Chunk中，被Top 5找回的比例。
- HitRate@5：Top 5中至少有1个相关Chunk的问题占比。
- MRR@5：第一个相关Chunk越靠前，分数越高。
- NDCG@5：综合评价多个相关Chunk的整体排序位置。

如果报告提示索引存在乱码或标注Chunk缺失，应先修复索引再把指标作为有效结论。
