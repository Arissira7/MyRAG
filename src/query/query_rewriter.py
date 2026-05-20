"""查询改写模块 — Query Rewrite。

面试应答要点:
Q: 为什么要做查询改写？
A: 用户提问往往是模糊的、口语化的，和文档中的表述有差异。
   例如用户问"这个集群能跑多大的模型"，文档里可能写的是
   "单节点最大支持 70B 参数模型的分布式训练"。
   查询改写通过同义词扩展、指代消解、问题分解来弥合这个语义鸿沟。

Q: 指代消解怎么做？
A: 用 LLM 识别当前 query 中指向上下文的部分（如"它"、"这个"），
   然后结合历史对话窗口，把指代还原为具体实体。
   例如: 问"A100的显存多大"，下一句问"它的功耗呢" →
   改写为"查询内容：A100的显存和功耗"。

Q: 子查询分解怎么做？
A: 把一个复杂问题拆成多个简单问题并行检索，再合并结果。
   例如"比较V100和A100在训练Transformer时的性能差异" →
   拆成 ["V100训练Transformer性能", "A100训练Transformer性能", "V100和A100对比"]，
   并行检索后取各路 top-3 的并集。
"""

from dataclasses import dataclass, field
from typing import Optional

from src.config import get_config


@dataclass
class RewrittenQuery:
    """查询改写结果。"""

    original_query: str
    rewritten_queries: list[str]
    intent: str
    history_aware: bool = False
    decomposed: bool = False


class QueryRewriter:
    """查询改写器。

    流程：历史上下文 → 指代消解 → 同义词扩展 → 子查询分解 → 最终查询列表
    """

    def __init__(
        self,
        model_name: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434",
        max_subqueries: int = 3,
        history_window: int = 5,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.max_subqueries = max_subqueries
        self.history_window = history_window
        self._history: list[dict] = []

    def rewrite(
        self,
        query: str,
        history: Optional[list[dict]] = None,
        intent_only: bool = False,
    ) -> RewrittenQuery:
        """执行完整的查询改写流程。

        Args:
            query: 原始用户 query
            history: 历史对话记录（用于指代消解）
            intent_only: 若为 True，只做意图识别，不做查询改写

        Returns:
            RewrittenQuery 包含改写后的查询列表和意图识别结果
        """
        if history:
            self._history = history[-self.history_window:]

        intent = self._identify_intent(query)

        if intent_only:
            return RewrittenQuery(
                original_query=query,
                rewritten_queries=[query],
                intent=intent,
            )

        history_aware_query = self._resolve_references(query)

        expanded_queries = self._expand_synonyms(history_aware_query)

        decomposed = self._decompose_query(expanded_queries)

        final_queries = decomposed if len(decomposed) > 1 else [history_aware_query]

        final_queries = self._deduplicate(final_queries)[: self.max_subqueries]

        return RewrittenQuery(
            original_query=query,
            rewritten_queries=final_queries,
            intent=intent,
            history_aware=len(self._history) > 0,
            decomposed=len(final_queries) > 1,
        )

    def _identify_intent(self, query: str) -> str:
        """识别查询意图类别。"""
        prompt = f"""请识别以下查询的意图类别。只输出一个词（事实型/推理型/比较型/定义型/操作型）：

查询: {query}

意图: """

        try:
            import requests
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 8},
                },
                timeout=10,
            )
            response.raise_for_status()
            return response.json().get("response", "事实型").strip()
        except Exception:
            return "事实型"

    def _resolve_references(self, query: str) -> str:
        """指代消解：把"它"、"这个"等指代还原为具体实体。"""
        if not self._history:
            return query

        context_text = self._build_history_context()

        prompt = f"""以下是对话历史和当前查询。请将当前查询中的指代词还原为具体实体。

历史对话:
{context_text}

当前查询: {query}

请直接输出改写后的查询（只输出改写后的文本，不要解释）："""

        try:
            import requests
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 200},
                },
                timeout=15,
            )
            response.raise_for_status()
            rewritten = response.json().get("response", "").strip()
            return rewritten if rewritten else query
        except Exception:
            return query

    def _build_history_context(self) -> str:
        """构建历史对话上下文。"""
        if not self._history:
            return ""
        parts = []
        for i, turn in enumerate(self._history[-3:]):
            parts.append(f"用户: {turn.get('query', '')}")
            parts.append(f"助手: {turn.get('answer', '')[:200]}")
        return "\n".join(parts)

    def _expand_synonyms(self, query: str) -> str:
        """同义词扩展——将口语化表达扩展为文档用语。"""
        synonym_map = {
            "跑": "运行",
            "训练": "训练",
            "卡": "GPU",
            "显存": "GPU显存",
            "集群": "计算集群",
            "机柜": "服务器机柜",
            "温度": "温控",
            "功耗": "TDP",
            "参数": "参数量",
        }

        expanded = query
        for informal, formal in synonym_map.items():
            if informal in expanded:
                expanded = expanded.replace(informal, f"{informal}/{formal}")

        return expanded

    def _decompose_query(self, query: str) -> list[str]:
        """分解复合问题为多个子查询。"""
        decompose_prompt = f"""请将以下复杂查询分解为多个简单的子查询。

要求：
1. 分解后的子查询应该能独立检索
2. 返回 1-3 个子查询，用 | 分隔
3. 如果查询本身很简单，直接返回原查询

查询: {query}

子查询（用 | 分隔）："""

        try:
            import requests
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": decompose_prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 100},
                },
                timeout=15,
            )
            response.raise_for_status()
            result = response.json().get("response", "").strip()

            sub_queries = [q.strip() for q in result.split("|")]
            sub_queries = [q for q in sub_queries if q]

            return sub_queries if sub_queries else [query]
        except Exception:
            return [query]

    def _deduplicate(self, queries: list[str]) -> list[str]:
        """去重近义查询。"""
        seen: set[str] = set()
        unique: list[str] = []
        for q in queries:
            normalized = q.lower().replace(" ", "")
            if normalized not in seen:
                seen.add(normalized)
                unique.append(q)
        return unique

    def add_to_history(self, query: str, answer: str) -> None:
        """将本次对话加入历史。"""
        self._history.append({"query": query, "answer": answer[:500]})
        if len(self._history) > self.history_window * 2:
            self._history = self._history[-self.history_window * 2:]

    def clear_history(self) -> None:
        """清空历史对话。"""
        self._history = []
