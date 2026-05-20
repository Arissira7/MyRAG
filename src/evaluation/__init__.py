"""RAG 评测模块。

提供完整的评测能力，包括：
- 评测集数据结构（BenchmarkItem）
- 指标计算（Recall@K, MRR, NDCG@K, HitRate@K）
- 多种 Baseline 方法对比
- 统计分析（置信区间、t-test）
- 完整评测流程（RAGEvaluator）
"""

from src.evaluation.evaluator import (
    # 评测集数据结构
    BenchmarkItem,
    RetrievalPrediction,
    IntentType,
    Difficulty,
    
    # 指标计算
    MetricsCalculator,
    
    # 评测集构造
    BenchmarkConstructor,
    
    # Baseline 方法
    RetrievalMethod,
    BaselineRetriever,
    
    # 统计分析
    StatisticalAnalyzer,
    
    # 完整评测流程
    RAGEvaluator,
    
    # 便捷函数
    load_benchmark,
    save_benchmark,
)

__all__ = [
    "BenchmarkItem",
    "RetrievalPrediction",
    "IntentType",
    "Difficulty",
    "MetricsCalculator",
    "BenchmarkConstructor",
    "RetrievalMethod",
    "BaselineRetriever",
    "StatisticalAnalyzer",
    "RAGEvaluator",
    "load_benchmark",
    "save_benchmark",
]
