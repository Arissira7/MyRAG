"""Cross-Encoder 重排序模块。

面试应答要点:
Q: 什么是 Cross-Encoder？和 Bi-Encoder 的区别？
A: Bi-Encoder 分别编码 query 和 document，得到两个独立向量。
   Cross-Encoder 把 (query, document) 拼接后一起编码，
   attention 机制可以看到 query 和 document 之间的 token 级别交互，
   精度更高，但速度更慢（无法预计算 document 向量）。

   应用场景：Bi-Encoder 做首轮粗召回（快速），Cross-Encoder 做精排（准确）。
   100万文档：Bi-Encoder 毫秒级检索 → Cross-Encoder 重排 top-100。

Q: Cross-Encoder 的训练目标是什么？
A: 二分类：给定 (query, doc) 对，输出 0-1 之间的相关性分数。
   正样本：(query, relevant_doc)，负样本：(query, irrelevant_doc)。
   我们使用本地 Qwen 模型做推理，不需要单独训练。

Q: 重排后取多少个？
A: 重排后取 top-10。这是因为：
   1. Cross-Encoder 计算代价高（需要逐对编码）
   2. top-10 之后分数梯度变化小，信息边际收益递减
   3. 太多 chunks 会给 LLM 造成上下文噪音
"""

from dataclasses import dataclass
from typing import Optional

from src.metadata import ChunkMetadata, RetrievedChunk


class CrossEncoderReranker:
    """Cross-Encoder 重排序器。

    使用 Qwen 本地模型对 (query, chunk) 做 pairwise 相关性评分，
    并按分数重新排序。
    """

    def __init__(
        self,
        model_name: str = "qwen2.5:1.5b",
        base_url: str = "http://localhost:11434",
        top_n: int = 10,
        batch_size: int = 8,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.top_n = top_n
        self.batch_size = batch_size
        self._prompt_template = self._build_prompt()

    def _build_prompt(self) -> str:
        """构建 Cross-Encoder 的评分 prompt。

        让模型对 (query, document) 给出一个 0-1 之间的相关性分数。
        """
        return """你是一个信息检索相关性评估模型。请评估以下查询和文档的相关程度。

查询: {query}

文档: {document}

请只输出一个 0 到 1 之间的小数分数，表示相关程度。0 表示完全不相关，1 表示完全相关。
只输出数字，不要输出其他内容。

分数: """

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_n: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        """对候选 chunks 进行重排序。

        Args:
            query: 查询文本
            chunks: 来自 hybrid_retriever 的候选 chunks（通常是 top-40）
            top_n: 返回重排后的 top-N（默认用 self.top_n）

        Returns:
            重排后的 RetrivedChunk 列表
        """
        if not chunks:
            return []

        n = top_n or self.top_n
        chunks_to_rerank = chunks[: min(len(chunks), self.top_n * 4)]

        scored_chunks = self._score_chunks(query, chunks_to_rerank)

        scored_chunks.sort(key=lambda x: x.score, reverse=True)

        for i, chunk in enumerate(scored_chunks[:n]):
            chunk.rank = i

        return scored_chunks[:n]

    def _score_chunks(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """使用 Cross-Encoder 对每个 chunk 评分。"""
        import requests

        scored = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            for chunk in batch:
                prompt = self._prompt_template.format(
                    query=query,
                    document=chunk.text[:2000],
                )

                try:
                    response = requests.post(
                        f"{self.base_url}/api/generate",
                        json={
                            "model": self.model_name,
                            "prompt": prompt,
                            "stream": False,
                            "options": {"temperature": 0.1, "num_predict": 8},
                        },
                        timeout=30,
                    )
                    response.raise_for_status()
                    result_text = response.json().get("response", "0").strip()

                    score = self._parse_score(result_text)
                    chunk.score = score
                    scored.append(chunk)

                except Exception as e:
                    chunk.score = 0.0
                    scored.append(chunk)

        return scored

    def _parse_score(self, text: str) -> float:
        """从模型输出中解析相关性分数。"""
        import re
        text = text.strip()

        match = re.search(r"0?\.\d+", text)
        if match:
            score = float(match.group())
            return min(1.0, max(0.0, score))

        try:
            score = float(text)
            return min(1.0, max(0.0, score))
        except ValueError:
            return 0.0

    def score_single(
        self,
        query: str,
        document: str,
    ) -> float:
        """对单个 (query, document) 对评分。"""
        temp_chunk = RetrievedChunk(
            chunk_id="temp",
            text=document,
            metadata=ChunkMetadata(
                tenant_id="",
                department="",
                doc_id="",
                doc_title="",
                chunk_id="temp",
            ),
            score=0.0,
        )
        scored = self._score_chunks(query, [temp_chunk])
        return scored[0].score if scored else 0.0
