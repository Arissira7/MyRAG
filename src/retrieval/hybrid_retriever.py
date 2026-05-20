"""多路召回与分数融合模块。

并行执行 BM25（稀疏）+ Dense（稠密）检索，通过分数归一化和加权融合
得到统一的候选列表。

面试应答要点:
Q: 为什么要混合检索？只用向量检索不行吗？
A: 向量检索对专有名词、型号编号的精确匹配能力弱。
   例如查询"A100和H800的区别"，向量检索可能召回"GPU型号对比"相关内容，
   但精确命中"A100"和"H800"这两个词的能力不如 BM25。
   混合检索让两种方法互补。

Q: 分数归一化怎么做？
A: 使用 min-max 归一化，把 BM25 分数和向量分数都映射到 [0,1] 区间。
   BM25 分数范围取决于文档集合，向量内积范围是 [-1,1]（归一化后为 [0,1]）。
   归一化后两者才能直接加权求和。

Q: alpha 参数怎么调？
A: alpha 控制 BM25 和向量检索的权重。
   - alpha=1.0：纯 BM25
   - alpha=0.0：纯向量检索
   - alpha=0.5：各占一半
   我们通过网格搜索（alpha ∈ {0.3, 0.4, 0.5, 0.6, 0.7}）
   在评测集上找最优值，最终落在 0.5 附近。

Q: RRF 融合是什么？
A: Reciprocal Rank Fusion (RRF) 是一种无参数排序融合方法。
   公式: RRF_score(d) = Σ 1/(k + rank_i(d))
   其中 k 是平滑参数（通常取 60），rank_i 是方法 i 中文档 d 的排名。
   RRF 的优点是不需要调参，且对不同尺度的分数不敏感。
   我们把它作为备选融合方案。

Q: 为什么召回 top-20 而不只是 top-5？
A: 第一轮召回要保证高召回率，宁可多召回一些让后面的重排模块来筛选。
   如果首轮召回就很少，重排后的效果上限也会受限。
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.chunker import Chunk
from src.metadata import ChunkMetadata, RetrievedChunk
from src.storage.bm25_index import BM25Indexer
from src.storage.vector_store import VectorStore, EmbeddingModel


@dataclass
class RetrievalResult:
    """一次完整的检索结果。"""

    query: str
    bm25_results: list[tuple[str, float, ChunkMetadata]]
    dense_results: list[tuple[str, float, ChunkMetadata]]
    fused_results: list[RetrievedChunk]
    used_reranker: bool = False


class HybridRetriever:
    """混合检索器 — BM25 + Dense Vector 并行召回 + 加权融合。

    检索流程:
    1. BM25 召回 top-20（专有名词匹配）
    2. Dense 召回 top-20（语义相似）
    3. 分数归一化
    4. 加权融合（或 RRF 融合）
    5. 输出融合后的候选列表
    """

    def __init__(
        self,
        bm25_indexer: BM25Indexer,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        top_k_bm25: int = 20,
        top_k_dense: int = 20,
        fusion_alpha: float = 0.5,
        rrf_k: int = 60,
        use_rrf: bool = False,
    ):
        self.bm25_indexer = bm25_indexer
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.top_k_bm25 = top_k_bm25
        self.top_k_dense = top_k_dense
        self.fusion_alpha = fusion_alpha
        self.rrf_k = rrf_k
        self.use_rrf = use_rrf

    def retrieve(
        self,
        query: str,
        tenant_id: Optional[str] = None,
        department: Optional[str] = None,
        doc_ids: Optional[list[str]] = None,
        top_k: int = 40,
    ) -> list[RetrievedChunk]:
        """执行混合检索。

        Args:
            query: 查询文本
            tenant_id: 租户过滤
            department: 部门过滤
            doc_ids: 文档ID过滤
            top_k: 返回的最终候选数量

        Returns:
            RetrivedChunk 列表，按融合分数降序排列
        """
        bm25_results = self.bm25_indexer.search(
            query=query,
            top_k=self.top_k_bm25,
            tenant_id=tenant_id,
            department=department,
            doc_ids=doc_ids,
        )

        query_vector = self.embedding_model.embed_single(query)
        dense_results = self.vector_store.search(
            query_vector=query_vector,
            top_k=self.top_k_dense,
            tenant_id=tenant_id,
            department=department,
            doc_ids=doc_ids,
        )

        fused = self._fuse_results(
            bm25_results=bm25_results,
            dense_results=dense_results,
            top_k=top_k,
        )

        return fused

    def _fuse_results(
        self,
        bm25_results: list[tuple[str, float, ChunkMetadata]],
        dense_results: list[tuple[str, float, ChunkMetadata]],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """融合 BM25 和向量检索结果。"""
        if self.use_rrf:
            return self._rrf_fusion(bm25_results, dense_results, top_k)
        return self._weighted_fusion(bm25_results, dense_results, top_k)

    def _weighted_fusion(
        self,
        bm25_results: list[tuple[str, float, ChunkMetadata]],
        dense_results: list[tuple[str, float, ChunkMetadata]],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """加权融合。"""
        bm25_scores = {cid: score for cid, score, _ in bm25_results}
        dense_scores = {cid: score for cid, score, _ in dense_results}

        all_ids = set(bm25_scores.keys()) | set(dense_scores.keys())

        if not bm25_scores and not dense_scores:
            return []

        bm25_max = max(bm25_scores.values()) if bm25_scores else 1.0
        dense_max = max(dense_scores.values()) if dense_scores else 1.0
        bm25_min = min(bm25_scores.values()) if bm25_scores else 0.0
        dense_min = min(dense_scores.values()) if dense_scores else 0.0

        fused_scores: dict[str, float] = {}
        for chunk_id in all_ids:
            bm25_s = bm25_scores.get(chunk_id, 0.0)
            dense_s = dense_scores.get(chunk_id, 0.0)

            bm25_norm = (bm25_s - bm25_min) / (bm25_max - bm25_min + 1e-8)
            dense_norm = (dense_s - dense_min) / (dense_max - dense_min + 1e-8)

            fused = self.fusion_alpha * bm25_norm + (1 - self.fusion_alpha) * dense_norm
            fused_scores[chunk_id] = fused

        sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)

        bm25_map = {cid: (score, meta) for cid, score, meta in bm25_results}
        dense_map = {cid: (score, meta) for cid, score, meta in dense_results}

        retrieved_chunks: list[RetrievedChunk] = []
        for rank, chunk_id in enumerate(sorted_ids[:top_k]):
            score = fused_scores[chunk_id]

            if chunk_id in bm25_map and chunk_id in dense_map:
                source = "hybrid"
                original_score = (bm25_map[chunk_id][0] + dense_map[chunk_id][0]) / 2
            elif chunk_id in bm25_map:
                source = "bm25"
                original_score = bm25_map[chunk_id][0]
            else:
                source = "dense"
                original_score = dense_map[chunk_id][0]

            meta = (bm25_map.get(chunk_id) or dense_map.get(chunk_id))[1]

            bm25_text = self.bm25_indexer.get_chunk_text(chunk_id)
            dense_text = self.vector_store.get_text(chunk_id)
            text = bm25_text or dense_text or ""

            retrieved_chunks.append(RetrievedChunk(
                chunk_id=chunk_id,
                text=text,
                metadata=meta,
                score=score,
                source=source,
                rank=rank,
            ))

        return retrieved_chunks

    def _rrf_fusion(
        self,
        bm25_results: list[tuple[str, float, ChunkMetadata]],
        dense_results: list[tuple[str, float, ChunkMetadata]],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Reciprocal Rank Fusion（无参数排序融合）。

        公式: RRF(d) = Σ 1/(k + rank(d))
        """
        rrf_scores: dict[str, float] = {}
        bm25_map: dict[str, tuple[float, ChunkMetadata]] = {}
        dense_map: dict[str, tuple[float, ChunkMetadata]] = {}

        for rank, (cid, score, meta) in enumerate(bm25_results):
            bm25_map[cid] = (score, meta)
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (self.rrf_k + rank + 1)

        for rank, (cid, score, meta) in enumerate(dense_results):
            dense_map[cid] = (score, meta)
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (self.rrf_k + rank + 1)

        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        retrieved_chunks: list[RetrievedChunk] = []
        for rank, chunk_id in enumerate(sorted_ids[:top_k]):
            if chunk_id in bm25_map and chunk_id in dense_map:
                source = "hybrid"
            elif chunk_id in bm25_map:
                source = "bm25"
            else:
                source = "dense"

            meta = (bm25_map.get(chunk_id) or dense_map.get(chunk_id))[1]

            bm25_text = self.bm25_indexer.get_chunk_text(chunk_id)
            dense_text = self.vector_store.get_text(chunk_id)
            text = bm25_text or dense_text or ""

            retrieved_chunks.append(RetrievedChunk(
                chunk_id=chunk_id,
                text=text,
                metadata=meta,
                score=rrf_scores[chunk_id],
                source=source,
                rank=rank,
            ))

        return retrieved_chunks
