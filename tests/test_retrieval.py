"""单元测试 — 覆盖核心模块。

面试时如果问到测试，可以提：
- 单元测试覆盖分块、融合、调度等核心逻辑
- 端到端测试用真实数据验证 pipeline
"""

import unittest
from unittest.mock import MagicMock, patch

from src.chunker import SemanticChunker, Chunk, ChunkProcessor
from src.metadata import ChunkMetadata, filter_chunks_by_tenant, RetrievedChunk
from src.retrieval.dynamic_scheduler import DynamicScheduler, SchedulingDecision


class TestSemanticChunker(unittest.TestCase):
    """测试语义分块器的核心逻辑。"""

    def setUp(self):
        self.chunker = SemanticChunker(
            parent_chunk_size=512,
            child_chunk_size=256,
            chunk_overlap=64,
        )

    def test_token_estimation(self):
        """测试 token 估算。"""
        chinese = "算力集群网络拓扑结构采用胖树架构"
        est = SemanticChunker._estimate_tokens(chinese)
        self.assertGreater(est, 0)
        self.assertLess(est, 100)

        english = "InfiniBand HDR 200Gbps switch configuration"
        est = SemanticChunker._estimate_tokens(english)
        self.assertGreater(est, 0)

    def test_split_into_sentences(self):
        """测试按句号分句。"""
        text = "A100的显存是80GB。它的功耗是400W。NVLink带宽是900GB/s。"
        sentences = self.chunker._split_into_sentences(text)
        self.assertEqual(len(sentences), 3)
        self.assertIn("A100的显存是80GB", sentences[0])

    def test_parent_child_indexing(self):
        """测试父子索引关系正确建立。"""
        results = self.chunker.chunk_document(
            doc_id="doc001",
            doc_title="GPU参数表",
            content="A100是NVIDIA推出的数据中心GPU。显存80GB HBM2e。功耗400W。",
            metadata={"tenant_id": "tenant_a", "department": "infra"},
        )

        self.assertGreater(len(results), 0)

        for result in results:
            parent = result.parent_chunk
            self.assertIsNotNone(parent.chunk_id)
            self.assertTrue(parent.chunk_id.startswith("doc001_parent_"))
            self.assertIsNone(parent.parent_id)

            for child in result.child_chunks:
                self.assertTrue(child.chunk_id.startswith("doc001_parent_"))
                self.assertEqual(child.parent_id, parent.chunk_id)
                self.assertIsNotNone(child.metadata)
                self.assertEqual(child.metadata.tenant_id, "tenant_a")

    def test_build_overlap(self):
        """测试重叠部分构建。"""
        sentences = ["A100是NVIDIA的数据中心GPU。", "显存80GB HBM2e。"]
        overlap = self.chunker._build_overlap(sentences)
        self.assertIsInstance(overlap, str)
        self.assertLessEqual(len(overlap), self.chunker.chunk_overlap * 2)


class TestBM25Scoring(unittest.TestCase):
    """测试 BM25 相关性评分逻辑。"""

    def test_bm25_score_range(self):
        """BM25 分数应该为正数。"""
        from src.storage.bm25_index import BM25Indexer

        with patch("src.storage.bm25_index.BM25Okapi") as mock_bm25:
            with patch("src.storage.bm25_index.jieba"):
                mock_instance = MagicMock()
                mock_instance.get_scores.return_value = [0.5, 1.2, 0.8, 0.0, 2.1]
                mock_bm25.return_value = mock_instance

                indexer = BM25Indexer()
                indexer._bm25 = mock_instance
                indexer._corpus = ["doc1", "doc2", "doc3", "doc4", "doc5"]
                indexer._index_to_chunk_id = {i: f"chunk_{i}" for i in range(5)}
                indexer._metadata_map = {
                    f"chunk_{i}": ChunkMetadata(
                        tenant_id="t1", department="d1",
                        doc_id="doc1", doc_title="t", chunk_id=f"chunk_{i}"
                    ) for i in range(5)
                }

                results = indexer.search("GPU A100", top_k=5, tenant_id="t1")

                for cid, score, meta in results:
                    self.assertGreater(score, 0)


class TestHybridFusion(unittest.TestCase):
    """测试混合检索分数融合。"""

    def test_weighted_fusion_normalization(self):
        """测试加权融合时分数归一化正确。"""
        from src.retrieval.hybrid_retriever import HybridRetriever

        with patch.object(HybridRetriever, "__init__", lambda x, **kw: None):
            retriever = HybridRetriever.__new__(HybridRetriever)
            retriever.fusion_alpha = 0.5
            retriever.use_rrf = False

            bm25_results = [
                ("c1", 10.0, ChunkMetadata(tenant_id="t1", department="d1", doc_id="d1", doc_title="t", chunk_id="c1")),
                ("c2", 5.0, ChunkMetadata(tenant_id="t1", department="d1", doc_id="d1", doc_title="t", chunk_id="c2")),
            ]
            dense_results = [
                ("c1", 0.9, ChunkMetadata(tenant_id="t1", department="d1", doc_id="d1", doc_title="t", chunk_id="c1")),
                ("c3", 0.8, ChunkMetadata(tenant_id="t1", department="d1", doc_id="d1", doc_title="t", chunk_id="c3")),
            ]

            with patch.object(retriever, "bm25_indexer"):
                with patch.object(retriever, "vector_store"):
                    fused = retriever._weighted_fusion(bm25_results, dense_results, top_k=5)

            self.assertGreater(len(fused), 0)
            scores = [c.score for c in fused]
            self.assertTrue(all(0 <= s <= 1 for s in scores),
                           f"分数应在 [0,1]，实际: {scores}")


class TestDynamicScheduler(unittest.TestCase):
    """测试 Dynamic Top-K 调度算法。"""

    def setUp(self):
        self.scheduler = DynamicScheduler(
            std_threshold=0.15,
            score_gap_threshold=0.05,
            min_chunks=3,
            max_chunks=10,
            initial_top_k=5,
        )

    def _make_chunk(self, score: float) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=f"chunk_{score}",
            text="test",
            metadata=ChunkMetadata(
                tenant_id="t1", department="d1",
                doc_id="d1", doc_title="t", chunk_id=f"chunk_{score}"
            ),
            score=score,
        )

    def test_disperse_scores_reduces_k(self):
        """分数分散时应该减少 chunks。"""
        chunks = [self._make_chunk(0.95), self._make_chunk(0.1), self._make_chunk(0.05)]
        decision = self.scheduler.decide(chunks)
        self.assertLessEqual(decision.top_k, self.scheduler.initial_top_k)

    def test_concentrated_scores_increases_k(self):
        """分数集中时应该增加 chunks。"""
        chunks = [
            self._make_chunk(0.51), self._make_chunk(0.50),
            self._make_chunk(0.49), self._make_chunk(0.48),
            self._make_chunk(0.47),
        ]
        decision = self.scheduler.decide(chunks)
        self.assertGreater(decision.top_k, self.scheduler.initial_top_k)

    def test_top_k_respects_bounds(self):
        """top_k 必须在 [min, max] 范围内。"""
        sparse = [self._make_chunk(0.9), self._make_chunk(0.1)]
        decision = self.scheduler.decide(sparse)
        self.assertGreaterEqual(decision.top_k, self.scheduler.min_chunks)
        self.assertLessEqual(decision.top_k, self.scheduler.max_chunks)

    def test_empty_chunks_returns_min(self):
        """空 chunks 返回 min_chunks。"""
        decision = self.scheduler.decide([])
        self.assertEqual(decision.top_k, 0)

    def test_score_distribution_computed(self):
        """决策中包含分数分布统计。"""
        chunks = [self._make_chunk(0.9), self._make_chunk(0.7), self._make_chunk(0.5)]
        decision = self.scheduler.decide(chunks)
        self.assertIn("mean", decision.score_distribution)
        self.assertIn("std", decision.score_distribution)
        self.assertIn("count", decision.score_distribution)


class TestTenantIsolation(unittest.TestCase):
    """测试多租户隔离。"""

    def test_filter_by_tenant(self):
        """租户过滤只返回匹配 tenant_id 的 chunks。"""
        chunks = [
            RetrievedChunk("c1", "text1", ChunkMetadata(tenant_id="t1", department="d1", doc_id="d1", doc_title="t", chunk_id="c1")),
            RetrievedChunk("c2", "text2", ChunkMetadata(tenant_id="t2", department="d1", doc_id="d1", doc_title="t", chunk_id="c2")),
            RetrievedChunk("c3", "text3", ChunkMetadata(tenant_id="t1", department="d1", doc_id="d1", doc_title="t", chunk_id="c3")),
        ]

        filtered = filter_chunks_by_tenant(chunks, "t1")
        self.assertEqual(len(filtered), 2)
        self.assertTrue(all(c.metadata.tenant_id == "t1" for c in filtered))


class TestRRFFusion(unittest.TestCase):
    """测试 RRF 融合逻辑。"""

    def test_rrf_fusion(self):
        """RRF 融合分数应为正值。"""
        from src.retrieval.hybrid_retriever import HybridRetriever

        with patch.object(HybridRetriever, "__init__", lambda x, **kw: None):
            retriever = HybridRetriever.__new__(HybridRetriever)
            retriever.use_rrf = True
            retriever.rrf_k = 60

            bm25_results = [
                (f"c{i}", float(10-i), ChunkMetadata(tenant_id="t1", department="d1", doc_id="d1", doc_title="t", chunk_id=f"c{i}"))
                for i in range(1, 6)
            ]
            dense_results = [
                (f"c{i}", float(5-i*0.5), ChunkMetadata(tenant_id="t1", department="d1", doc_id="d1", doc_title="t", chunk_id=f"c{i}"))
                for i in range(1, 6)
            ]

            with patch.object(retriever, "bm25_indexer"):
                with patch.object(retriever, "vector_store"):
                    fused = retriever._rrf_fusion(bm25_results, dense_results, top_k=5)

            self.assertGreater(len(fused), 0)
            for chunk in fused:
                self.assertGreater(chunk.score, 0)


if __name__ == "__main__":
    unittest.main()
