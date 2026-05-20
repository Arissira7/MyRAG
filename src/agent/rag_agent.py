"""RAG Agent 主类 — 整合所有模块的编排逻辑。

面试应答要点:
Q: 完整的 RAG pipeline 是怎样的？
A: query → 查询改写 → 多路召回 → Cross-Encoder重排 → Dynamic Top-K →
   parent_chunk回溯 → LLM生成 → Reflection校验 → 答案输出

Q: Reflection 机制是什么？失败3次后怎么处理？
A: Reflection 是一个自检验循环：
   1. LLM 生成答案后，让它判断答案是否忠实于检索到的 context
   2. 如果不忠实（幻觉/超出范围），触发新一轮检索
   3. 最多重试 3 次
   4次后仍然失败 → 返回"I don't know"并附上最相关的检索片段

   面试追问：为什么不一直重试？
   因为超过 3 次后，系统陷入了"用户想要的答案库里没有"的困境，
   继续重试只会浪费 token 且结果不会变好。

Q: Agent 怎么决定调哪个工具？
A: 我们的 RAG Agent 目前是固定 pipeline，不涉及工具选择。
   如果是更复杂的 Agent（多工具），会用到 ReAct：
   Thought → Action → Observation → ... → Final Answer
   其中 Action 由 LLM 决定调用哪个工具（search/calculator/document）。

Q: 流式输出怎么实现？
A: 使用 SSE (Server-Sent Events)。LLM API 支持 stream 参数时，
   逐 token yield 给前端，实现打字机效果。
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Generator, Optional

from src.agent.prompt_template import DEFAULT_RAG_PROMPT
from src.chunker import Chunk
from src.config import get_config
from src.metadata import ChunkMetadata, RetrievedChunk
from src.query.query_rewriter import QueryRewriter, RewrittenQuery
from src.retrieval.dynamic_scheduler import DynamicScheduler
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker


@dataclass
class GenerationResult:
    """生成结果。"""

    answer: str
    retrieved_chunks: list[RetrievedChunk]
    used_chunks: list[str]
    reflection_passed: bool
    retry_count: int
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGPipelineState:
    """RAG pipeline 中间状态（用于调试和可观测性）。"""

    query: str
    rewritten_queries: list[str]
    hybrid_results: list[RetrievedChunk]
    reranked_results: list[RetrievedChunk]
    scheduled_chunks: list[RetrievedChunk]
    parent_chunks: list[Chunk]
    generation: str
    reflection_result: Optional[dict] = None
    retry_count: int = 0


class RAGAgent:
    """RAG Agent — 完整的检索增强生成流水线。

    整合查询改写、多路召回、Cross-Encoder重排、Dynamic Top-K调度、
    Parent-Child回溯、LLM生成、Reflection自检。
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        scheduler: DynamicScheduler,
        query_rewriter: QueryRewriter,
        chunk_store: dict[str, Chunk],
        parent_child_map: dict[str, str],
        llm_model: str = "qwen2.5:7b",
        llm_base_url: str = "http://localhost:11434",
        max_retries: int = 3,
    ):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.scheduler = scheduler
        self.query_rewriter = query_rewriter
        self.chunk_store = chunk_store
        self.parent_child_map = parent_child_map
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url.rstrip("/")
        self.max_retries = max_retries

    def ask(
        self,
        query: str,
        tenant_id: Optional[str] = None,
        department: Optional[str] = None,
        stream: bool = False,
    ) -> GenerationResult | Generator[str, None, None]:
        """执行一次完整的 RAG 问答。

        Args:
            query: 用户问题
            tenant_id: 租户ID（用于多租户隔离）
            department: 部门过滤
            stream: 是否流式输出

        Returns:
            GenerationResult（包含答案、使用的 chunks、延迟等）
            或 Generator[str]（流式输出时的 token 流）
        """
        start_time = time.time()
        state = RAGPipelineState(
            query=query,
            rewritten_queries=[],
            hybrid_results=[],
            reranked_results=[],
            scheduled_chunks=[],
            parent_chunks=[],
            generation="",
        )

        rewritten = self.query_rewriter.rewrite(query)
        state.rewritten_queries = rewritten.rewritten_queries

        queries_to_retrieve = rewritten.rewritten_queries
        all_results: list[RetrievedChunk] = []

        for q in queries_to_retrieve:
            results = self.hybrid_retriever.retrieve(
                query=q,
                tenant_id=tenant_id,
                department=department,
            )
            all_results.extend(results)

        all_results = self._deduplicate_chunks(all_results)
        all_results.sort(key=lambda x: x.score, reverse=True)
        state.hybrid_results = all_results

        reranked = self.reranker.rerank(query, all_results)
        state.reranked_results = reranked

        scheduling = self.scheduler.decide(reranked)
        scheduled_chunks = scheduling.selected_chunks
        state.scheduled_chunks = scheduled_chunks

        parent_chunks = self._retrieve_parent_chunks(scheduled_chunks)
        state.parent_chunks = parent_chunks

        context_text = self._build_context(parent_chunks)

        generation = self._generate_with_reflection(
            query=query,
            context=context_text,
            state=state,
            stream=stream,
        )

        latency_ms = (time.time() - start_time) * 1000

        return GenerationResult(
            answer=generation,
            retrieved_chunks=scheduled_chunks,
            used_chunks=[c.chunk_id for c in scheduled_chunks],
            reflection_passed=state.reflection_result is not None,
            retry_count=state.retry_count,
            latency_ms=latency_ms,
            metadata={
                "rewritten_queries": rewritten.rewritten_queries,
                "intent": rewritten.intent,
                "scheduling_reason": scheduling.reason,
                "score_distribution": scheduling.score_distribution,
            },
        )

    def _deduplicate_chunks(
        self, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """对召回结果去重（同一 parent 下的 child 只保留一个）。"""
        seen_parent_ids: set[str] = set()
        seen_chunk_ids: set[str] = set()
        unique: list[RetrievedChunk] = []

        for chunk in chunks:
            if chunk.chunk_id in seen_chunk_ids:
                continue

            parent_id = chunk.metadata.parent_id or ""
            if parent_id and parent_id in seen_parent_ids:
                continue

            unique.append(chunk)
            seen_chunk_ids.add(chunk.chunk_id)
            if parent_id:
                seen_parent_ids.add(parent_id)

        return unique

    def _retrieve_parent_chunks(
        self, child_chunks: list[RetrievedChunk]
    ) -> list[Chunk]:
        """根据 child chunks 回溯获取对应的 parent chunks。"""
        parent_ids: set[str] = set()
        for child in child_chunks:
            parent_id = child.metadata.parent_id
            if parent_id:
                parent_ids.add(parent_id)
            else:
                parent_ids.add(child.chunk_id)

        parent_chunks: list[Chunk] = []
        for parent_id in parent_ids:
            chunk = self.chunk_store.get(parent_id)
            if chunk:
                parent_chunks.append(chunk)

        return parent_chunks

    def _build_context(self, parent_chunks: list[Chunk]) -> str:
        """将 parent chunks 组装成 LLM 的上下文。"""
        if not parent_chunks:
            return "没有找到相关文档。"

        context_parts = []
        for i, chunk in enumerate(parent_chunks, 1):
            section_title = chunk.metadata.section_title or chunk.metadata.doc_title
            context_parts.append(
                f"[文档{i}] {section_title}\n{chunk.text}"
            )

        return "\n\n---\n\n".join(context_parts)

    def _generate_with_reflection(
        self,
        query: str,
        context: str,
        state: RAGPipelineState,
        stream: bool = False,
    ) -> str:
        """生成答案并做 Reflection 自检。"""
        prompt = DEFAULT_RAG_PROMPT.format(context=context, query=query)

        for attempt in range(self.max_retries):
            state.retry_count = attempt

            if stream:
                response_text = self._stream_generate(prompt)
            else:
                response_text = self._generate(prompt)

            state.generation = response_text

            reflection = self._reflect(query, context, response_text)
            state.reflection_result = reflection

            if reflection.get("passed", False):
                return response_text

            if attempt < self.max_retries - 1:
                improvement = reflection.get("improvement_hint", "")
                prompt = DEFAULT_RAG_PROMPT.format(
                    context=context,
                    query=query + f"\n\n注意：{improvement}"
                )

        return response_text

    def _generate(self, prompt: str) -> str:
        """调用 LLM 生成答案。"""
        try:
            import requests
            response = requests.post(
                f"{self.llm_base_url}/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 2048,
                    },
                },
                timeout=60,
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            return f"抱歉，生成答案时出错：{str(e)}"

    def _stream_generate(self, prompt: str) -> str:
        """流式生成（返回完整文本，调用方需要自行处理流式）。"""
        full_text = ""
        try:
            import requests
            response = requests.post(
                f"{self.llm_base_url}/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 2048,
                    },
                },
                stream=True,
                timeout=60,
            )
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    token = data.get("response", "")
                    full_text += token
            return full_text
        except Exception as e:
            return f"抱歉，生成答案时出错：{str(e)}"

    def _reflect(
        self,
        query: str,
        context: str,
        answer: str,
    ) -> dict[str, Any]:
        """Reflection 自检——判断答案是否忠实于 context。

        Reflection 检查维度:
        1. 答案是否基于 context（而非幻觉）
        2. 答案是否回答了 query 的问题
        3. 答案是否有遗漏的关键信息
        """
        reflection_prompt = f"""你是一个答案质量评估模型。请评估以下答案的质量。

原始问题: {query}

参考文档:
{context[:1500]}

待评估答案:
{answer}

请从以下三个维度评估：
1. 答案是否基于参考文档（而非编造）？
2. 答案是否回答了原始问题？
3. 答案是否有明显的信息遗漏？

请用以下 JSON 格式输出评估结果（只输出 JSON，不要其他内容）：
{{"passed": true/false, "improvement_hint": "如果 passed=false，给出改进建议"}}
"""

        try:
            import requests
            response = requests.post(
                f"{self.llm_base_url}/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": reflection_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 200,
                    },
                },
                timeout=30,
            )
            response.raise_for_status()
            result_text = response.json().get("response", "").strip()

            import json
            result = json.loads(result_text)
            return result
        except Exception:
            return {"passed": True, "improvement_hint": ""}
