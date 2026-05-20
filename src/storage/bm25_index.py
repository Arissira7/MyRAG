"""BM25 稀疏向量检索模块。

基于 rank_bm25 库实现，对中文使用 jieba 分词。
支持 metadata 过滤以实现多租户隔离。

面试应答要点:
Q: BM25 适合什么场景？
A: BM25 擅长精确匹配专有名词、术语、型号、编号。
   例如"GPU A100"、"V100"、"A800"的精确匹配，
   这些词在语义空间中可能因为训练数据偏差而距离很远，
   但 BM25 能通过词项匹配精确命中。

Q: BM25 的原理？
A: BM25 是一种基于词项频率的信息检索模型，是 TF-IDF 的改良版本。
   核心公式: Score(d,q) = IDF(t) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * |d|/avgdl))
   其中 k1 控制词频饱和度（避免词频过高时分数无限增长），
   b 控制文档长度归一化（短文档更容易获得高分）。
   我们使用 k1=1.5, b=0.75 的经典参数。

Q: 分词怎么处理中英混合？
A: jieba 对中文分词，对英文按空格和标点分词。
   例如"GPU A100训练集群" → ["GPU", "A100", "训练", "集群"]
"""

import pickle
from pathlib import Path
from typing import Optional

try:
    import jieba
except ImportError:
    jieba = None

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

from src.chunker import Chunk
from src.metadata import ChunkMetadata


class BM25Indexer:
    """BM25 索引构建与检索。

    构建流程: tokenized_corpus → BM25Okapi → 持久化
    检索流程: query → tokenize → BM25.search → 结果过滤
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        tenant_field: str = "tenant_id",
    ):
        self.k1 = k1
        self.b = b
        self.tenant_field = tenant_field
        self._corpus: list[str] = []
        self._tokenized_corpus: list[list[str]] = []
        self._chunk_id_to_index: dict[str, int] = {}
        self._index_to_chunk_id: dict[int, str] = {}
        self._metadata_map: dict[str, ChunkMetadata] = {}
        self._bm25: Optional[BM25Okapi] = None

    def build_index(self, chunks: list[Chunk]) -> None:
        """构建 BM25 索引。

        Args:
            chunks: Chunk 列表（同时包含 parent 和 child chunks）
        """
        if BM25Okapi is None or jieba is None:
            raise ImportError("请安装依赖: pip install rank-bm25 jieba")

        self._corpus = []
        self._tokenized_corpus = []
        self._chunk_id_to_index = {}
        self._metadata_map = {}

        for idx, chunk in enumerate(chunks):
            chunk_text = chunk.text.strip()
            if not chunk_text:
                continue

            self._corpus.append(chunk_text)
            self._tokenized_corpus.append(self._tokenize(chunk_text))
            self._chunk_id_to_index[chunk.chunk_id] = idx
            self._index_to_chunk_id[idx] = chunk.chunk_id

            if chunk.metadata:
                self._metadata_map[chunk.chunk_id] = chunk.metadata

        self._bm25 = BM25Okapi(self._tokenized_corpus, k1=self.k1, b=self.b)

    def _tokenize(self, text: str) -> list[str]:
        """中英文混合分词。"""
        text = text.lower()
        import re
        tokens: list[str] = []

        for segment in re.split(r"([a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)*)", text):
            if re.match(r"^[a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)*$", segment):
                tokens.extend(self._tokenize_english(segment))
            else:
                tokens.extend(self._tokenize_chinese(segment))

        return tokens

    def _tokenize_chinese(self, text: str) -> list[str]:
        """中文分词。"""
        if jieba is None:
            return list(text)
        return [t for t in jieba.cut(text) if t.strip() and len(t) > 0]

    def _tokenize_english(self, text: str) -> list[str]:
        """英文分词（按子词切分以处理专业术语）。"""
        import re
        parts = re.split(r"[.\-_/]", text)
        result: list[str] = []
        for part in parts:
            part = part.lower().strip()
            if len(part) >= 2:
                result.append(part)
            elif part:
                result.append(part)
        return result if result else [text.lower()]

    def search(
        self,
        query: str,
        top_k: int = 20,
        tenant_id: Optional[str] = None,
        department: Optional[str] = None,
        doc_ids: Optional[list[str]] = None,
    ) -> list[tuple[str, float, ChunkMetadata]]:
        """检索相关 chunks。

        Args:
            query: 查询文本
            top_k: 返回数量
            tenant_id: 租户ID过滤（多租户隔离）
            department: 部门过滤
            doc_ids: 指定文档ID列表过滤

        Returns:
            [(chunk_id, bm25_score, metadata)] 按分数降序排列
        """
        if self._bm25 is None:
            raise RuntimeError("索引未构建，请先调用 build_index()")

        query_tokens = self._tokenize(query)
        scores = self._bm25.get_scores(query_tokens)

        results: list[tuple[str, float, ChunkMetadata]] = []
        for idx, score in enumerate(scores):
            if score <= 0:
                continue
            chunk_id = self._index_to_chunk_id.get(idx)
            if not chunk_id:
                continue

            metadata = self._metadata_map.get(chunk_id)
            if not metadata:
                continue

            if not self._passes_filter(metadata, tenant_id, department, doc_ids):
                continue

            results.append((chunk_id, float(score), metadata))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _passes_filter(
        self,
        metadata: ChunkMetadata,
        tenant_id: Optional[str],
        department: Optional[str],
        doc_ids: Optional[list[str]],
    ) -> bool:
        """检查 metadata 是否通过过滤条件。"""
        if tenant_id and metadata.tenant_id != tenant_id:
            return False
        if department and metadata.department != department:
            return False
        if doc_ids and metadata.doc_id not in doc_ids:
            return False
        return True

    def save(self, path: str) -> None:
        """持久化索引到文件。"""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "corpus": self._corpus,
            "tokenized_corpus": self._tokenized_corpus,
            "chunk_id_to_index": self._chunk_id_to_index,
            "index_to_chunk_id": self._index_to_chunk_id,
            "metadata_map": {k: v.to_dict() for k, v in self._metadata_map.items()},
            "k1": self.k1,
            "b": self.b,
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

        if self._bm25 is not None:
            self._bm25.save(str(save_path.with_suffix(".index")))

    def load(self, path: str) -> None:
        """从文件加载索引。"""
        with open(path, "rb") as f:
            state = pickle.load(f)

        self._corpus = state["corpus"]
        self._tokenized_corpus = state["tokenized_corpus"]
        self._chunk_id_to_index = state["chunk_id_to_index"]
        self._index_to_chunk_id = state["index_to_chunk_id"]
        self._metadata_map = {
            k: ChunkMetadata.from_dict(v) for k, v in state["metadata_map"].items()
        }
        self.k1 = state["k1"]
        self.b = state["b"]

        index_path = Path(path).with_suffix(".index")
        if index_path.exists():
            self._bm25 = BM25Okapi.load(str(index_path), self._tokenized_corpus)
        else:
            self._bm25 = BM25Okapi(self._tokenized_corpus, k1=self.k1, b=self.b)

    def get_chunk_text(self, chunk_id: str) -> Optional[str]:
        """根据 chunk_id 获取原文。"""
        idx = self._chunk_id_to_index.get(chunk_id)
        if idx is not None and idx < len(self._corpus):
            return self._corpus[idx]
        return None
