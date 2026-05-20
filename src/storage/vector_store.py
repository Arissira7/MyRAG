"""向量存储模块 — 基于 Qwen3-Embedding + FAISS。

面试应答要点:
Q: 为什么用 FAISS？
A: FAISS 是 Facebook Research 开源的向量检索库，适合亿级向量规模下的
   高效最近邻搜索。我们使用 IndexFlatIP（内积）做精确检索，
   对于更大规模数据（>100万向量）可以切换到 IVF-PQ 倒排索引加速。

Q: 为什么用内积而不是余弦相似度？
A: Qwen3-Embedding 输出已经做了 L2 归一化（normalize=True），
   归一化后的向量内积等价于余弦相似度。这样既有内积的计算效率，
   又保持了余弦相似度的尺度不变性。

Q: 向量维度怎么选？
A: Qwen3-Embedding 输出 1024 维向量。我们直接使用原始维度，
   没有做 PCA 降维——因为降维会损失信息，尤其是对于领域术语。

Q: 向量检索和 BM25 各适合什么场景？
A: 向量检索擅长语义匹配：同义词（"GPU" vs "显卡"）、
   语义相似但词项不同的问题（"机器学习训练需要什么硬件" vs "深度学习用什么卡"）。
   BM25 擅长精确匹配：型号编号（"A100"、"H800"）、专有术语（"InfiniBand"、"NVLink"）。
   两者互补，所以要做混合检索。

Q: 怎么处理多租户？
A: 在检索时传入 tenant_id filter，FAISS 会先在过滤后的子空间搜索。
   实现方式：预过滤（构建tenant专属索引）或后过滤（全局搜索后过滤）。
   我们使用后过滤 + BM25双保险，确保隔离性。
"""

import json
import pickle
from pathlib import Path
from typing import Any, Optional

try:
    import faiss
except ImportError:
    faiss = None

from src.chunker import Chunk
from src.metadata import ChunkMetadata


class EmbeddingModel:
    """Embedding 模型封装 — 支持 Ollama 本地部署的 Qwen3-Embedding。

    通过 Ollama API 调用本地模型，避免 API 费用。
    """

    def __init__(
        self,
        model_name: str = "qwen3-embedding:latest",
        base_url: str = "http://localhost:11434",
        dimension: int = 1024,
        batch_size: int = 32,
        normalize: bool = True,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.dimension = dimension
        self.batch_size = batch_size
        self.normalize = normalize

    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量生成文本的向量表示。

        通过 Ollama API 调用 Qwen3-Embedding。
        API: POST /api/embeddings
        Request: {"model": "qwen3-embedding", "prompt": "..."}
        Response: {"embedding": [...]}
        """
        try:
            import requests
        except ImportError:
            raise ImportError("请安装 requests: pip install requests")

        embeddings: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = []

            for text in batch:
                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model_name, "prompt": text},
                    timeout=30,
                )
                response.raise_for_status()
                result = response.json()
                embedding = result.get("embedding", [])

                if self.normalize:
                    import math
                    norm = math.sqrt(sum(x * x for x in embedding))
                    if norm > 0:
                        embedding = [x / norm for x in embedding]

                batch_embeddings.append(embedding)

            embeddings.extend(batch_embeddings)

        return embeddings

    def embed_single(self, text: str) -> list[float]:
        """生成单个文本的向量。"""
        return self.embed([text])[0]


class VectorStore:
    """向量存储 — FAISS IndexFlatIP + 完整 metadata 映射。

    使用内积（IP）作为相似度度量，配合归一化向量等价于余弦相似度。
    """

    def __init__(self, dimension: int = 1024):
        if faiss is None:
            raise ImportError("请安装 faiss-cpu 或 faiss-gpu: pip install faiss-cpu")

        self.dimension = dimension
        self._index: Optional[Any] = None
        self._chunk_ids: list[str] = []
        self._metadata_map: dict[str, ChunkMetadata] = {}
        self._text_map: dict[str, str] = {}

    def build_index(self, chunks: list[Chunk], embedding_model: EmbeddingModel) -> None:
        """构建向量索引。

        Args:
            chunks: Chunk 列表
            embedding_model: Embedding 模型实例
        """
        if not chunks:
            return

        texts = [chunk.text for chunk in chunks]
        chunk_ids = [chunk.chunk_id for chunk in chunks]

        print(f"正在为 {len(chunks)} 个 chunks 生成向量...")
        embeddings = embedding_model.embed(texts)

        self._index = faiss.IndexFlatIP(self.dimension)
        import numpy as np
        embeddings_array = np.array(embeddings, dtype=np.float32)
        self._index.add(embeddings_array)

        self._chunk_ids = chunk_ids
        for chunk in chunks:
            self._metadata_map[chunk.chunk_id] = chunk.metadata
            self._text_map[chunk.chunk_id] = chunk.text

        print(f"向量索引构建完成: {self._index.ntotal} 条向量")

    def search(
        self,
        query_vector: list[float],
        top_k: int = 20,
        tenant_id: Optional[str] = None,
        department: Optional[str] = None,
        doc_ids: Optional[list[str]] = None,
        score_threshold: float = 0.0,
    ) -> list[tuple[str, float, ChunkMetadata]]:
        """向量最近邻搜索。

        Args:
            query_vector: 查询向量
            top_k: 返回数量
            tenant_id: 租户过滤
            department: 部门过滤
            doc_ids: 文档ID过滤
            score_threshold: 分数阈值过滤

        Returns:
            [(chunk_id, score, metadata)] 按分数降序
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        import numpy as np
        query_np = np.array([query_vector], dtype=np.float32)

        search_k = min(top_k * 3, int(self._index.ntotal))
        distances, indices = self._index.search(query_np, search_k)

        results: list[tuple[str, float, ChunkMetadata]] = []
        seen_ids: set[str] = set()

        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._chunk_ids):
                continue

            chunk_id = self._chunk_ids[idx]
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)

            metadata = self._metadata_map.get(chunk_id)
            if not metadata:
                continue

            if not self._passes_filter(metadata, tenant_id, department, doc_ids):
                continue

            score = float(dist)
            if score < score_threshold:
                continue

            results.append((chunk_id, score, metadata))

            if len(results) >= top_k:
                break

        return results

    def _passes_filter(
        self,
        metadata: ChunkMetadata,
        tenant_id: Optional[str],
        department: Optional[str],
        doc_ids: Optional[list[str]],
    ) -> bool:
        """检查过滤条件。"""
        if tenant_id and metadata.tenant_id != tenant_id:
            return False
        if department and metadata.department != department:
            return False
        if doc_ids and metadata.doc_id not in doc_ids:
            return False
        return True

    def get_text(self, chunk_id: str) -> Optional[str]:
        """根据 chunk_id 获取原文。"""
        return self._text_map.get(chunk_id)

    def save(self, index_path: str, meta_path: str) -> None:
        """持久化索引。"""
        if self._index is not None:
            faiss.write_index(self._index, index_path)

        meta = {
            "chunk_ids": self._chunk_ids,
            "metadata_map": {k: v.to_dict() for k, v in self._metadata_map.items()},
            "text_map": self._text_map,
            "dimension": self.dimension,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def load(self, index_path: str, meta_path: str) -> None:
        """加载索引。"""
        self._index = faiss.read_index(index_path)

        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        self._chunk_ids = meta["chunk_ids"]
        self._metadata_map = {
            k: ChunkMetadata.from_dict(v) for k, v in meta["metadata_map"].items()
        }
        self._text_map = meta["text_map"]
        self.dimension = meta["dimension"]
