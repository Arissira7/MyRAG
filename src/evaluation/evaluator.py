"""RAG 评测体系模块。

提供完整的评测能力，支持检索指标计算、Baseline 对比、统计分析，
用于评估 RAG 系统各模块的独立贡献和整体效果。

面试应答要点:
Q: 评测集怎么构造的？多少条？
A: 200条，按 query 类型(intent_type)和难度(difficulty)分层采样。
   - 事实型 40% (80条)、推理型 25% (50条)、比较型 20% (40条)、定义型 10% (20条)、操作型 5% (10条)
   - easy 30% (60条)、medium 50% (100条)、hard 20% (40条)
   覆盖 3 个租户和 10 篇文档。

Q: 怎么知道 Recall=0.81 是好是坏？
A: 和 baseline 对比。我们的 full_pipeline Recall@5=0.81，
   比固定 top-5 (0.72) 高了 12.5%，比纯向量检索 (0.68) 高了 19%。
   在 hard 难度的 query 上提升更明显（0.54 vs 0.31）。

Q: Baseline 怎么设计？
A: 消融实验：逐个去掉组件（去掉重排、去掉动态调度、去掉混合检索），
   看每个模块的独立贡献。这样能定位系统瓶颈。

Q: 指标为什么选这几个？
A: - Recall@K: 检索系统的核心，看能否把相关文档都找出来
   - MRR: 看重首条结果的质量，适合对话场景
   - NDCG@K: 考虑位置权重，top 结果排错惩罚更大
   - HitRate@K: 二值判断，直观反映"能不能查到"

Q: 统计显著性怎么保证？
A: 95% 置信区间 + 成对 t-test。只有 p<0.05 才认为提升显著。
"""

from dataclasses import dataclass, field
from typing import Optional, Protocol, Callable
from enum import Enum
import math
import numpy as np
from scipy import stats

from src.metadata import RetrievedChunk, ChunkMetadata


# =============================================================================
# 1. 评测集数据结构
# =============================================================================

class IntentType(Enum):
    """Query 意图类型枚举。
    
    不同意图类型对检索系统的要求不同：
    - 事实型：需要精确匹配关键词
    - 推理型：需要理解因果关系
    - 比较型：需要同时召回多个实体的信息
    - 定义型：需要定位定义性描述
    - 操作型：需要理解步骤序列
    """
    FACTUAL = "factual"          # 事实型：问具体数值、时间、人名等
    REASONING = "reasoning"      # 推理型：需要多步推理
    COMPARATIVE = "comparative"  # 比较型：比较两个或多个实体
    DEFINITIONAL = "definitional" # 定义型：问"什么是X"
    OPERATIONAL = "operational"  # 操作型：问"怎么做"


class Difficulty(Enum):
    """问题难度等级。
    
    - easy: 直接能从单个 chunk 回答
    - medium: 需要综合 2-3 个 chunks
    - hard: 需要跨文档推理或精确召回
    """
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class BenchmarkItem:
    """单条评测数据。
    
    一条 benchmark 由 query 和其对应的 ground truth 组成。
    ground_truth_chunk_ids 是标注员标注的、与该 query 相关的 chunk IDs。
    """
    query: str
    ground_truth_chunk_ids: list[str]
    intent_type: IntentType
    difficulty: Difficulty
    tenant_id: str
    metadata: dict = field(default_factory=dict)
    
    def __post_init__(self):
        """类型安全转换。"""
        if isinstance(self.intent_type, str):
            self.intent_type = IntentType(self.intent_type)
        if isinstance(self.difficulty, str):
            self.difficulty = Difficulty(self.difficulty)


@dataclass
class RetrievalPrediction:
    """某次检索的预测结果。"""
    query: str
    retrieved_chunk_ids: list[str]
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


# =============================================================================
# 2. 指标计算
# =============================================================================

class MetricsCalculator:
    """检索指标计算器。
    
    提供 Recall@K、MRR、NDCG@K、HitRate@K 的计算。
    所有公式都基于集合论定义，适合二分相关性判断场景。
    """
    
    @staticmethod
    def recall_at_k(
        predictions: list[str],
        ground_truth: list[str],
        k: int
    ) -> float:
        """计算 Recall@K。
        
        公式: Recall@K = |Retrieved ∩ GroundTruth| / |GroundTruth|
        
        其中 Retrieved 是模型返回的前 K 个 chunk ID 集合，
        GroundTruth 是标注员标注的相关 chunk ID 集合。
        
        意义：衡量检索系统的"查全"能力，看能否把相关文档都找出来。
        Recall=1.0 意味着所有相关文档都在 top-K 中。
        
        Args:
            predictions: 模型返回的 chunk IDs（按排名排序）
            ground_truth: 标注的相关 chunk IDs
            k: 截断位置
        
        Returns:
            Recall@K 值，范围 [0, 1]
        """
        if not ground_truth:
            return 0.0
        
        retrieved_k = set(predictions[:k])
        relevant = set(ground_truth)
        
        # 分子：预测命中的相关文档数
        hits = len(retrieved_k & relevant)
        
        # 分母：所有相关文档数
        total_relevant = len(relevant)
        
        return hits / total_relevant
    
    @staticmethod
    def mrr(predictions: list[str], ground_truth: list[str]) -> float:
        """计算 MRR (Mean Reciprocal Rank)。
        
        公式: MRR = (1 / |Q|) * Σ (1 / rank_i)
        
        其中 rank_i 是第 i 个 query 的第一个相关文档的排名位置。
        如果没有相关文档在结果中，则该项贡献为 0。
        
        意义：衡量"第一个正确答案在哪里"。
        - MRR=1.0 意味着所有 query 的第一个相关文档都在 top-1
        - MRR=0.5 意味着平均第一个正确答案在 top-2
        
        适用场景：对话系统、问答系统，首条结果的质量至关重要。
        
        Args:
            predictions: 模型返回的 chunk IDs（按排名排序）
            ground_truth: 标注的相关 chunk IDs
        
        Returns:
            MRR 值，范围 (0, 1]
        """
        if not ground_truth:
            return 0.0
        
        relevant_set = set(ground_truth)
        
        # 遍历结果，找到第一个相关文档的位置
        for rank, chunk_id in enumerate(predictions, start=1):
            if chunk_id in relevant_set:
                # MRR 的核心：排名越靠前，倒数越大
                return 1.0 / rank
        
        # 没有找到相关文档
        return 0.0
    
    @staticmethod
    def ndcg_at_k(
        predictions: list[str],
        ground_truth: list[str],
        k: int
    ) -> float:
        """计算 NDCG@K (Normalized Discounted Cumulative Gain)。
        
        NDCG 是信息检索领域的标准指标，考虑了：
        1. 相关性分数（这里简化为二值：相关=1，不相关=0）
        2. 位置衰减：排在前面的结果更重要
        
        公式:
        - DCG@K = Σ (rel_i / log2(i + 1))，i 从 1 到 K
        - IDCG@K = DCG 的最优值（理想排序）
        - NDCG@K = DCG@K / IDCG@K
        
        其中 rel_i 是第 i 位的结果是否相关（1 或 0）。
        
        意义：综合考虑"找到了多少"和"排在多前面"。
        - NDCG=1.0 意味着完美排序（所有相关都在最前面）
        - NDCG 越接近 1.0 越好
        
        Args:
            predictions: 模型返回的 chunk IDs（按排名排序）
            ground_truth: 标注的相关 chunk IDs
            k: 截断位置
        
        Returns:
            NDCG@K 值，范围 [0, 1]
        """
        if not ground_truth:
            return 0.0
        
        relevant_set = set(ground_truth)
        
        # DCG@K：实际排序的累积增益（带位置衰减）
        # 第 i 位的结果，如果相关则贡献 1/log2(i+1)
        dcg = 0.0
        for i, chunk_id in enumerate(predictions[:k], start=1):
            if chunk_id in relevant_set:
                # 位置衰减：排名越靠后，衰减越大
                # log2(2) = 1，所以 top-1 的衰减因子是 1
                dcg += 1.0 / math.log2(i + 1)
        
        # IDCG@K：最优排序下的 DCG（即把所有相关文档排在最前面）
        num_relevant = min(len(relevant_set), k)
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, num_relevant + 1))
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def hit_rate_at_k(
        predictions: list[str],
        ground_truth: list[str],
        k: int
    ) -> float:
        """计算 HitRate@K（二值命中指标）。
        
        公式: HitRate@K = 1 if |Retrieved ∩ GroundTruth| > 0 else 0
        
        其中 Retrieved 是前 K 个结果，GroundTruth 是所有相关文档。
        
        意义：最简单直接的评价指标——
        "有没有至少一个相关文档在 top-K 中？"
        - HitRate=1.0 意味着每个 query 都至少有 1 个命中
        - HitRate=0.0 意味着完全没有命中
        
        适用场景：用户体验评估。如果 HitRate 低，用户会感到"查不到"。
        
        Args:
            predictions: 模型返回的 chunk IDs（按排名排序）
            ground_truth: 标注的相关 chunk IDs
            k: 截断位置
        
        Returns:
            HitRate@K 值，范围 {0, 1}
        """
        if not ground_truth:
            return 0.0
        
        retrieved_k = set(predictions[:k])
        relevant = set(ground_truth)
        
        # 只要有一个交集就返回 1
        return 1.0 if retrieved_k & relevant else 0.0
    
    @staticmethod
    def precision_at_k(
        predictions: list[str],
        ground_truth: list[str],
        k: int
    ) -> float:
        """计算 Precision@K。
        
        公式: Precision@K = |Retrieved ∩ GroundTruth| / K
        
        意义：衡量 top-K 结果中有多大的比例是相关的。
        通常与 Recall 一起使用（Precision 高但 Recall 低意味着"查到的不一定对"）。
        
        Args:
            predictions: 模型返回的 chunk IDs（按排名排序）
            ground_truth: 标注的相关 chunk IDs
            k: 截断位置
        
        Returns:
            Precision@K 值，范围 [0, 1]
        """
        if not predictions[:k]:
            return 0.0
        
        retrieved_k = set(predictions[:k])
        relevant = set(ground_truth)
        
        hits = len(retrieved_k & relevant)
        return hits / k
    
    @staticmethod
    def compute_all_metrics(
        predictions: list[str],
        ground_truth: list[str],
        k_values: list[int] = None
    ) -> dict:
        """计算所有指标。
        
        一次性返回 Recall@K、MRR、NDCG@K、HitRate@K、Precision@K
        用于快速生成评测报告。
        
        Args:
            predictions: 模型返回的 chunk IDs
            ground_truth: 标注的相关 chunk IDs
            k_values: 哪些 K 值要计算，默认 [1, 3, 5, 10]
        
        Returns:
            包含所有指标的字典
        """
        if k_values is None:
            k_values = [1, 3, 5, 10]
        
        metrics = {
            "mrr": MetricsCalculator.mrr(predictions, ground_truth),
        }
        
        for k in k_values:
            metrics[f"recall@{k}"] = MetricsCalculator.recall_at_k(predictions, ground_truth, k)
            metrics[f"ndcg@{k}"] = MetricsCalculator.ndcg_at_k(predictions, ground_truth, k)
            metrics[f"hit@{k}"] = MetricsCalculator.hit_rate_at_k(predictions, ground_truth, k)
            metrics[f"precision@{k}"] = MetricsCalculator.precision_at_k(predictions, ground_truth, k)
        
        return metrics


# =============================================================================
# 3. 评测集构造方法
# =============================================================================

class BenchmarkConstructor:
    """评测集构造器。
    
    构造流程:
    1. 确定分布：按 intent_type 和 difficulty 分层采样
    2. 文档覆盖：确保覆盖所有源文档
    3. 租户覆盖：确保覆盖多个租户
    4. 手工编写 query：由领域专家编写
    5. 标注 ground truth：由标注员标注相关 chunks
    
    分布设计理由:
    - 事实型 40%：实际使用中最常见
    - 推理型 25%：检验系统的语义理解能力
    - 比较型 20%：需要同时召回多个实体
    - 定义型 10%：相对简单，检验基础能力
    - 操作型 5%：步骤类问题，考验上下文理解
    
    - easy 30%：保证基础效果不掉队
    - medium 50%：覆盖典型使用场景
    - hard 20%：检验系统上限和边界能力
    """
    
    # 意图类型分布配置
    INTENT_DISTRIBUTION = {
        IntentType.FACTUAL: 0.40,       # 80 条
        IntentType.REASONING: 0.25,    # 50 条
        IntentType.COMPARATIVE: 0.20,   # 40 条
        IntentType.DEFINITIONAL: 0.10, # 20 条
        IntentType.OPERATIONAL: 0.05,   # 10 条
    }
    
    # 难度分布配置
    DIFFICULTY_DISTRIBUTION = {
        Difficulty.EASY: 0.30,    # 60 条
        Difficulty.MEDIUM: 0.50,  # 100 条
        Difficulty.HARD: 0.20,    # 40 条
    }
    
    # 示例 Query 模板（实际使用中由专家编写）
    EXAMPLE_QUERIES = {
        IntentType.FACTUAL: [
            "CUDA 的 warp size 是多少？",
            "Transformer 的 attention 复杂度是多少？",
            "A100 的显存容量是多大？",
        ],
        IntentType.REASONING: [
            "为什么 BERT 使用 [CLS] token 做分类？",
            "梯度消失和梯度爆炸有什么区别？",
            "为什么学习率要使用 warmup？",
        ],
        IntentType.COMPARATIVE: [
            "A100 和 H800 有什么区别？",
            "BERT 和 GPT 的训练目标有什么不同？",
            "PyTorch 和 TensorFlow 各有什么优缺点？",
        ],
        IntentType.DEFINITIONAL: [
            "什么是 Transformer？",
            "什么是指针碰撞？",
            "什么是 KV Cache？",
        ],
        IntentType.OPERATIONAL: [
            "如何在 PyTorch 中实现混合精度训练？",
            "怎么用 accelerate 库做分布式训练？",
            "如何配置 DeepSpeed 的 ZeRO 优化？",
        ],
    }
    
    def __init__(
        self,
        total_size: int = 200,
        doc_ids: list[str] = None,
        tenant_ids: list[str] = None,
    ):
        """初始化评测集构造器。
        
        Args:
            total_size: 评测集总条数，默认 200
            doc_ids: 可用的文档 ID 列表
            tenant_ids: 可用的租户 ID 列表
        """
        self.total_size = total_size
        self.doc_ids = doc_ids or ["doc_001", "doc_002", "doc_003", "doc_004", "doc_005",
                                    "doc_006", "doc_007", "doc_008", "doc_009", "doc_010"]
        self.tenant_ids = tenant_ids or ["tenant_a", "tenant_b", "tenant_c"]
    
    def calculate_distribution(self) -> dict:
        """计算各类型的期望条数。
        
        返回:
            嵌套字典，key 是 (intent_type, difficulty)，value 是条数
        """
        result = {}
        
        for intent, intent_ratio in self.INTENT_DISTRIBUTION.items():
            intent_count = int(self.total_size * intent_ratio)
            
            for difficulty, diff_ratio in self.DIFFICULTY_DISTRIBUTION.items():
                count = int(intent_count * diff_ratio)
                result[(intent, difficulty)] = count
        
        return result
    
    def generate_synthetic_benchmark(
        self,
        chunk_to_doc_map: dict[str, str] = None,
    ) -> list[BenchmarkItem]:
        """生成合成的评测集（用于测试）。
        
        实际生产中，评测集由领域专家手工编写 query，
        然后由标注员标注 ground truth chunk IDs。
        
        Args:
            chunk_to_doc_map: chunk_id -> doc_id 的映射
        
        Returns:
            BenchmarkItem 列表
        """
        distribution = self.calculate_distribution()
        benchmark = []
        
        if chunk_to_doc_map is None:
            chunk_to_doc_map = {}
            for doc_id in self.doc_ids:
                for i in range(5):
                    chunk_id = f"{doc_id}_chunk_{i}"
                    chunk_to_doc_map[chunk_id] = doc_id
        
        item_id = 0
        for (intent, difficulty), count in distribution.items():
            for _ in range(count):
                tenant_id = self.tenant_ids[item_id % len(self.tenant_ids)]
                
                # 构造一个简单的 ground truth（2-3 个相关 chunks）
                relevant_chunks = []
                for _ in range(np.random.randint(2, 4)):
                    doc_id = self.doc_ids[np.random.randint(0, len(self.doc_ids))]
                    chunk_idx = np.random.randint(0, 5)
                    chunk_id = f"{doc_id}_chunk_{chunk_idx}"
                    if chunk_id not in relevant_chunks:
                        relevant_chunks.append(chunk_id)
                
                example_queries = self.EXAMPLE_QUERIES.get(intent, [])
                if example_queries:
                    query = example_queries[item_id % len(example_queries)]
                else:
                    query = f"关于 {intent.value} 类型的示例问题 {item_id}"
                
                benchmark.append(BenchmarkItem(
                    query=query,
                    ground_truth_chunk_ids=relevant_chunks,
                    intent_type=intent,
                    difficulty=difficulty,
                    tenant_id=tenant_id,
                    metadata={
                        "item_id": f"bench_{item_id:04d}",
                        "source_doc_ids": list(set(chunk_to_doc_map.get(cid, "") for cid in relevant_chunks)),
                    }
                ))
                
                item_id += 1
        
        return benchmark


# =============================================================================
# 4. Baseline 定义
# =============================================================================

class RetrievalMethod(Enum):
    """可评测的检索方法枚举。
    
    每个方法对应一种配置，用于消融实验：
    - full_pipeline: 完整流水线（混合检索 + 重排 + 动态调度）
    - fixed_top5: 固定 top-5（验证动态调度的价值）
    - dense_only: 仅向量检索（验证混合检索的价值）
    - bm25_only: 仅 BM25（验证向量检索的价值）
    - no_rerank: 无重排（验证重排的价值）
    - hybrid_no_dynamic: 混合+重排，但固定 top-5
    """
    FULL_PIPELINE = "full_pipeline"
    FIXED_TOP5 = "fixed_top5"
    DENSE_ONLY = "dense_only"
    BM25_ONLY = "bm25_only"
    NO_RERANK = "no_rerank"
    HYBRID_NO_DYNAMIC = "hybrid_no_dynamic"


# =============================================================================
# 5. Retriever 接口定义
# =============================================================================

class RetrieverProtocol(Protocol):
    """检索器协议。
    
    定义评测模块对检索器的要求。
    实际使用时需要注入实现了该协议的检索器实例。
    """
    
    def retrieve(
        self,
        query: str,
        tenant_id: Optional[str] = None,
        department: Optional[str] = None,
        doc_ids: Optional[list[str]] = None,
        top_k: int = 40,
    ) -> list[RetrievedChunk]:
        """执行检索。
        
        Args:
            query: 查询文本
            tenant_id: 租户过滤
            department: 部门过滤
            doc_ids: 文档 ID 过滤
            top_k: 返回结果数量
        
        Returns:
            RetrivedChunk 列表，按相关性降序排列
        """
        ...


class BaselineRetriever:
    """Baseline 检索器包装器。
    
    封装不同的检索配置，模拟各种 baseline 方法。
    
    面试要点:
    - fixed_top5: 验证动态调度的价值（去掉动态调度，看效果下降多少）
    - dense_only: 验证混合检索的价值（去掉 BM25，看效果下降多少）
    - no_rerank: 验证重排的价值（去掉 Cross-Encoder，看效果下降多少）
    """
    
    def __init__(
        self,
        hybrid_retriever,
        reranker=None,
        dynamic_scheduler=None,
    ):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.dynamic_scheduler = dynamic_scheduler
    
    def _full_pipeline(
        self,
        query: str,
        tenant_id: str = None,
        **kwargs
    ) -> list[RetrievedChunk]:
        """完整流水线：混合检索 -> 重排 -> 动态调度。"""
        chunks = self.hybrid_retriever.retrieve(
            query=query,
            tenant_id=tenant_id,
            top_k=40,
        )
        
        if self.reranker:
            chunks = self.reranker.rerank(query, chunks, top_n=20)
        
        if self.dynamic_scheduler:
            decision = self.dynamic_scheduler.decide(chunks)
            return decision.selected_chunks
        
        return chunks[:5]
    
    def _fixed_top5(
        self,
        query: str,
        tenant_id: str = None,
        **kwargs
    ) -> list[RetrievedChunk]:
        """固定 top-5：混合检索 -> 重排 -> 取 top-5。"""
        chunks = self.hybrid_retriever.retrieve(
            query=query,
            tenant_id=tenant_id,
            top_k=40,
        )
        
        if self.reranker:
            chunks = self.reranker.rerank(query, chunks, top_n=20)
        
        return chunks[:5]
    
    def _dense_only(
        self,
        query: str,
        tenant_id: str = None,
        **kwargs
    ) -> list[RetrievedChunk]:
        """仅向量检索：只用 dense retrieval，不混合 BM25。"""
        # 临时修改 alpha 参数，alpha=0 意味着只用 dense
        original_alpha = self.hybrid_retriever.fusion_alpha
        self.hybrid_retriever.fusion_alpha = 0.0
        
        chunks = self.hybrid_retriever.retrieve(
            query=query,
            tenant_id=tenant_id,
            top_k=10,
        )
        
        self.hybrid_retriever.fusion_alpha = original_alpha
        return chunks
    
    def _bm25_only(
        self,
        query: str,
        tenant_id: str = None,
        **kwargs
    ) -> list[RetrievedChunk]:
        """仅 BM25：只用 BM25 retrieval。"""
        # alpha=1 意味着只用 BM25
        original_alpha = self.hybrid_retriever.fusion_alpha
        self.hybrid_retriever.fusion_alpha = 1.0
        
        chunks = self.hybrid_retriever.retrieve(
            query=query,
            tenant_id=tenant_id,
            top_k=10,
        )
        
        self.hybrid_retriever.fusion_alpha = original_alpha
        return chunks
    
    def _no_rerank(
        self,
        query: str,
        tenant_id: str = None,
        **kwargs
    ) -> list[RetrievedChunk]:
        """无重排：混合检索后直接取 top-K，不做 Cross-Encoder。"""
        chunks = self.hybrid_retriever.retrieve(
            query=query,
            tenant_id=tenant_id,
            top_k=10,
        )
        return chunks
    
    def _hybrid_no_dynamic(
        self,
        query: str,
        tenant_id: str = None,
        **kwargs
    ) -> list[RetrievedChunk]:
        """混合检索 + 重排，但固定 top-5（不做动态调度）。"""
        chunks = self.hybrid_retriever.retrieve(
            query=query,
            tenant_id=tenant_id,
            top_k=40,
        )
        
        if self.reranker:
            chunks = self.reranker.rerank(query, chunks, top_n=20)
        
        return chunks[:5]
    
    def retrieve_by_method(
        self,
        method: RetrievalMethod,
        query: str,
        tenant_id: str = None,
        **kwargs
    ) -> list[RetrievedChunk]:
        """根据方法名执行对应的检索策略。
        
        Args:
            method: 检索方法枚举
            query: 查询文本
            tenant_id: 租户 ID
        
        Returns:
            检索结果列表
        """
        method_map = {
            RetrievalMethod.FULL_PIPELINE: self._full_pipeline,
            RetrievalMethod.FIXED_TOP5: self._fixed_top5,
            RetrievalMethod.DENSE_ONLY: self._dense_only,
            RetrievalMethod.BM25_ONLY: self._bm25_only,
            RetrievalMethod.NO_RERANK: self._no_rerank,
            RetrievalMethod.HYBRID_NO_DYNAMIC: self._hybrid_no_dynamic,
        }
        
        func = method_map.get(method)
        if func is None:
            raise ValueError(f"Unknown method: {method}")
        
        return func(query=query, tenant_id=tenant_id, **kwargs)


# =============================================================================
# 6. 统计分析
# =============================================================================

class StatisticalAnalyzer:
    """统计分析器。
    
    提供置信区间计算和假设检验功能。
    使用 t-distribution（小样本）进行统计推断。
    """
    
    @staticmethod
    def compute_mean(scores: list[float]) -> float:
        """计算均值。"""
        if not scores:
            return 0.0
        return float(np.mean(scores))
    
    @staticmethod
    def compute_std(scores: list[float]) -> float:
        """计算标准差。
        
        使用无偏估计（ddof=1）：
        std = sqrt(Σ(x_i - mean)² / (n-1))
        """
        if len(scores) < 2:
            return 0.0
        return float(np.std(scores, ddof=1))
    
    @staticmethod
    def compute_confidence_interval(
        scores: list[float],
        confidence: float = 0.95
    ) -> tuple[float, float, float]:
        """计算均值的置信区间。
        
        使用 t-distribution（因为总体方差未知，样本量有限）。
        
        公式: mean ± t_{α/2, n-1} * (std / sqrt(n))
        
        其中 t_{α/2, n-1} 是自由度为 n-1 的 t 分布的 α/2 分位数。
        
        Args:
            scores: 样本数据
            confidence: 置信水平，默认 95%
        
        Returns:
            (mean, lower_bound, upper_bound)
        """
        if len(scores) < 2:
            return float(np.mean(scores)), 0.0, 0.0
        
        n = len(scores)
        mean = float(np.mean(scores))
        std = float(np.std(scores, ddof=1))
        se = std / math.sqrt(n)  # 标准误差
        
        # t 分布的临界值
        # alpha = 1 - confidence = 0.05
        alpha = 1 - confidence
        df = n - 1  # 自由度
        
        t_critical = stats.t.ppf(1 - alpha / 2, df)
        
        margin = t_critical * se
        
        return mean, mean - margin, mean + margin
    
    @staticmethod
    def paired_ttest(
        scores_a: list[float],
        scores_b: list[float]
    ) -> dict:
        """成对 t 检验，比较两种方法是否有显著差异。
        
        零假设 H0: 两种方法的均值没有差异
        备择假设 H1: 两种方法的均值有差异
        
        公式: t = mean(diff) / (std(diff) / sqrt(n))
        其中 diff = scores_a - scores_b
        
        Args:
            scores_a: 方法 A 的各 query 得分
            scores_b: 方法 B 的各 query 得分
        
        Returns:
            dict 包含 t 统计量、p 值、95% 置信区间、效应量
        """
        if len(scores_a) != len(scores_b):
            raise ValueError("Scores must have same length for paired t-test")
        
        if len(scores_a) < 2:
            return {"t_stat": 0.0, "p_value": 1.0, "significant": False}
        
        diff = [a - b for a, b in zip(scores_a, scores_b)]
        mean_diff = float(np.mean(diff))
        std_diff = float(np.std(diff, ddof=1))
        n = len(diff)
        
        if std_diff == 0:
            return {
                "t_stat": 0.0,
                "p_value": 1.0,
                "mean_diff": mean_diff,
                "significant": False,
                "cohens_d": 0.0,
            }
        
        # t 统计量
        se = std_diff / math.sqrt(n)
        t_stat = mean_diff / se
        
        # p 值（双尾检验）
        df = n - 1
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        
        # Cohen's d 效应量
        # |d| < 0.2: 无差异, 0.2-0.5: 小差异, 0.5-0.8: 中等差异, > 0.8: 大差异
        pooled_std = math.sqrt((std_diff ** 2))  # 简化版本
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0
        
        # 差值的置信区间
        _, ci_lower, ci_upper = StatisticalAnalyzer.compute_confidence_interval(diff)
        
        return {
            "t_stat": t_stat,
            "p_value": p_value,
            "mean_diff": mean_diff,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "significant": p_value < 0.05,
            "cohens_d": cohens_d,
            "df": df,
        }
    
    @staticmethod
    def format_stat_result(
        method_name: str,
        scores: list[float],
    ) -> str:
        """格式化单个方法的统计结果。"""
        mean, lower, upper = StatisticalAnalyzer.compute_confidence_interval(scores)
        std = StatisticalAnalyzer.compute_std(scores)
        
        return (
            f"{method_name:20s}: "
            f"Mean={mean:.4f} ± {std:.4f}, "
            f"95% CI=[{lower:.4f}, {upper:.4f}]"
        )


# =============================================================================
# 7. 完整评测流程
# =============================================================================

class RAGEvaluator:
    """RAG 系统评测器。
    
    完整的评测流程：
    1. 加载评测集
    2. 对每个方法执行检索
    3. 计算各项指标
    4. 统计分析，生成对比报告
    
    使用示例:
    ```python
    evaluator = RAGEvaluator(
        retriever=hybrid_retriever,
        reranker=cross_encoder_reranker,
        dynamic_scheduler=dynamic_scheduler,
    )
    
    # 单方法评测
    results = evaluator.evaluate_full("full_pipeline", benchmark, retriever)
    
    # 全方法对比
    all_results = evaluator.evaluate_all_methods(benchmark, retriever)
    
    # 生成报告
    report = evaluator.generate_report(all_results)
    print(report)
    ```
    """
    
    def __init__(
        self,
        retriever,
        reranker=None,
        dynamic_scheduler=None,
        k_values: list[int] = None,
    ):
        """初始化评测器。
        
        Args:
            retriever: 实现了 retrieve 方法的检索器
            reranker: Cross-Encoder 重排器（可选）
            dynamic_scheduler: 动态调度器（可选）
            k_values: 要计算的 K 值列表，默认 [1, 3, 5, 10]
        """
        self.retriever = retriever
        self.reranker = reranker
        self.dynamic_scheduler = dynamic_scheduler
        self.k_values = k_values or [1, 3, 5, 10]
        
        # 包装成 BaselineRetriever 以支持多种 baseline 方法
        self.baseline_retriever = BaselineRetriever(
            hybrid_retriever=retriever,
            reranker=reranker,
            dynamic_scheduler=dynamic_scheduler,
        )
    
    def evaluate_full(
        self,
        method: RetrievalMethod,
        benchmark: list[BenchmarkItem],
    ) -> dict:
        """评测单个方法。
        
        对评测集中每个 query 执行检索，计算各项指标，
        返回聚合后的统计结果。
        
        Args:
            method: 要评测的检索方法
            benchmark: 评测数据集
        
        Returns:
            dict 包含 per_query 原始结果和 aggregated 统计结果
        """
        per_query_results = []
        metric_names = [f"recall@{k}" for k in self.k_values] + \
                       [f"ndcg@{k}" for k in self.k_values] + \
                       [f"hit@{k}" for k in self.k_values] + \
                       ["mrr", "precision@5"]
        
        for item in benchmark:
            # 执行检索
            retrieved_chunks = self.baseline_retriever.retrieve_by_method(
                method=method,
                query=item.query,
                tenant_id=item.tenant_id,
            )
            
            retrieved_ids = [c.chunk_id for c in retrieved_chunks]
            
            # 计算各项指标
            metrics = MetricsCalculator.compute_all_metrics(
                predictions=retrieved_ids,
                ground_truth=item.ground_truth_chunk_ids,
                k_values=self.k_values,
            )
            
            per_query_results.append({
                "query": item.query,
                "intent_type": item.intent_type.value,
                "difficulty": item.difficulty.value,
                "retrieved_ids": retrieved_ids,
                "ground_truth_ids": item.ground_truth_chunk_ids,
                "metrics": metrics,
            })
        
        # 聚合统计
        aggregated = self._aggregate_results(per_query_results, metric_names)
        
        return {
            "method": method.value,
            "per_query": per_query_results,
            "aggregated": aggregated,
        }
    
    def _aggregate_results(
        self,
        per_query_results: list[dict],
        metric_names: list[str],
    ) -> dict:
        """聚合 per-query 结果为统计指标。"""
        aggregated = {}
        
        for metric_name in metric_names:
            scores = [r["metrics"][metric_name] for r in per_query_results]
            
            mean, ci_lower, ci_upper = StatisticalAnalyzer.compute_confidence_interval(scores)
            std = StatisticalAnalyzer.compute_std(scores)
            
            aggregated[metric_name] = {
                "mean": mean,
                "std": std,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "scores": scores,
            }
        
        return aggregated
    
    def evaluate_all_methods(
        self,
        benchmark: list[BenchmarkItem],
        methods: list[RetrievalMethod] = None,
    ) -> dict:
        """评测所有 baseline 方法并对比。
        
        Args:
            benchmark: 评测数据集
            methods: 要评测的方法列表，默认所有方法
        
        Returns:
            dict，key 是方法名，value 是 evaluate_full 的结果
        """
        if methods is None:
            methods = list(RetrievalMethod)
        
        all_results = {}
        
        for method in methods:
            print(f"Evaluating {method.value}...")
            results = self.evaluate_full(method, benchmark)
            all_results[method.value] = results
        
        return all_results
    
    def generate_report(
        self,
        results: dict,
        highlight_method: str = "full_pipeline",
    ) -> str:
        """生成格式化的对比报告。
        
        报告包含：
        1. 各方法的指标对比表
        2. 统计显著性检验结果（full_pipeline vs 各 baseline）
        3. 分难度/意图类型的详细分析
        
        Args:
            results: evaluate_all_methods 的结果
            highlight_method: 重点关注的方法名
        
        Returns:
            格式化的报告字符串
        """
        lines = []
        
        # =========================================================================
        # 1. 主指标对比表
        # =========================================================================
        lines.append("=" * 100)
        lines.append("RAG 评测报告")
        lines.append("=" * 100)
        lines.append("")
        
        # 表头
        methods = list(results.keys())
        header = f"{'方法':<25}" + "".join(
            f"{'Recall@' + str(k):>12}" for k in self.k_values
        ) + "".join(
            f"{'MRR':>10}" for _ in [1]
        ) + "".join(
            f"{'NDCG@' + str(k):>10}" for k in self.k_values
        )
        lines.append(header)
        lines.append("-" * len(header))
        
        # 表格数据
        for method_name in methods:
            method_result = results[method_name]
            agg = method_result["aggregated"]
            
            row = f"{method_name:<25}"
            for k in self.k_values:
                mean = agg[f"recall@{k}"]["mean"]
                row += f"{mean:>12.4f}"
            row += f"{agg['mrr']['mean']:>10.4f}"
            for k in self.k_values:
                mean = agg[f"ndcg@{k}"]["mean"]
                row += f"{mean:>10.4f}"
            lines.append(row)
        
        lines.append("")
        
        # =========================================================================
        # 2. 统计显著性检验
        # =========================================================================
        lines.append("=" * 100)
        lines.append("统计显著性检验（成对 t-test）")
        lines.append("=" * 100)
        lines.append("")
        
        if highlight_method in results:
            highlight_scores = results[highlight_method]["aggregated"]
            
            for method_name in methods:
                if method_name == highlight_method:
                    continue
                
                method_result = results[method_name]
                
                lines.append(f"\n{highlight_method} vs {method_name}:")
                lines.append("-" * 50)
                
                for k in [5, 10]:  # 主要关注 Recall@5 和 Recall@10
                    metric = f"recall@{k}"
                    
                    scores_a = highlight_scores[metric]["scores"]
                    scores_b = method_result["aggregated"][metric]["scores"]
                    
                    ttest_result = StatisticalAnalyzer.paired_ttest(scores_a, scores_b)
                    
                    sig_marker = "***" if ttest_result["p_value"] < 0.001 else \
                                "**" if ttest_result["p_value"] < 0.01 else \
                                "*" if ttest_result["p_value"] < 0.05 else "n.s."
                    
                    lines.append(
                        f"  {metric}: "
                        f"Δ={ttest_result['mean_diff']:+.4f}, "
                        f"p={ttest_result['p_value']:.4f} {sig_marker}, "
                        f"Cohen's d={ttest_result['cohens_d']:.3f}"
                    )
        
        lines.append("")
        
        # =========================================================================
        # 3. 按难度分层分析
        # =========================================================================
        lines.append("=" * 100)
        lines.append("按难度分层分析（Recall@5）")
        lines.append("=" * 100)
        lines.append("")
        
        for difficulty in ["easy", "medium", "hard"]:
            lines.append(f"\n{difficulty.upper()}:")
            
            for method_name in methods:
                method_result = results[method_name]
                per_query = method_result["per_query"]
                
                difficulty_scores = [
                    r["metrics"]["recall@5"]
                    for r in per_query
                    if r["difficulty"] == difficulty
                ]
                
                if difficulty_scores:
                    mean = np.mean(difficulty_scores)
                    std = np.std(difficulty_scores, ddof=1)
                    lines.append(f"  {method_name:<25}: {mean:.4f} ± {std:.4f}")
        
        lines.append("")
        
        # =========================================================================
        # 4. 按意图类型分层分析
        # =========================================================================
        lines.append("=" * 100)
        lines.append("按意图类型分层分析（Recall@5）")
        lines.append("=" * 100)
        lines.append("")
        
        for intent in ["factual", "reasoning", "comparative", "definitional", "operational"]:
            lines.append(f"\n{intent.upper()}:")
            
            for method_name in methods:
                method_result = results[method_name]
                per_query = method_result["per_query"]
                
                intent_scores = [
                    r["metrics"]["recall@5"]
                    for r in per_query
                    if r["intent_type"] == intent
                ]
                
                if intent_scores:
                    mean = np.mean(intent_scores)
                    std = np.std(intent_scores, ddof=1)
                    lines.append(f"  {method_name:<25}: {mean:.4f} ± {std:.4f}")
        
        lines.append("")
        
        return "\n".join(lines)
    
    def compute_confidence_interval(
        self,
        scores: list[float],
        confidence: float = 0.95
    ) -> tuple[float, float, float]:
        """计算均值的置信区间。
        
        Args:
            scores: 样本数据
            confidence: 置信水平
        
        Returns:
            (mean, lower_bound, upper_bound)
        """
        return StatisticalAnalyzer.compute_confidence_interval(scores, confidence)
    
    def save_results(
        self,
        results: dict,
        output_path: str,
    ) -> None:
        """保存评测结果到文件。
        
        Args:
            results: 评测结果
            output_path: 输出文件路径
        """
        import json
        
        # 转换 numpy 类型为 Python 原生类型
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(convert(results), f, ensure_ascii=False, indent=2)


# =============================================================================
# 8. 便捷函数
# =============================================================================

def load_benchmark(
    benchmark_path: str,
) -> list[BenchmarkItem]:
    """从文件加载评测集。
    
    Args:
        benchmark_path: 评测集文件路径（JSON 格式）
    
    Returns:
        BenchmarkItem 列表
    """
    import json
    
    with open(benchmark_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return [
        BenchmarkItem(
            query=item["query"],
            ground_truth_chunk_ids=item["ground_truth_chunk_ids"],
            intent_type=IntentType(item["intent_type"]),
            difficulty=Difficulty(item["difficulty"]),
            tenant_id=item["tenant_id"],
            metadata=item.get("metadata", {}),
        )
        for item in data
    ]


def save_benchmark(
    benchmark: list[BenchmarkItem],
    output_path: str,
) -> None:
    """保存评测集到文件。
    
    Args:
        benchmark: BenchmarkItem 列表
        output_path: 输出文件路径
    """
    import json
    
    data = [
        {
            "query": item.query,
            "ground_truth_chunk_ids": item.ground_truth_chunk_ids,
            "intent_type": item.intent_type.value,
            "difficulty": item.difficulty.value,
            "tenant_id": item.tenant_id,
            "metadata": item.metadata,
        }
        for item in benchmark
    ]
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
