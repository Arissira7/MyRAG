# 部门智能 RAG 问答助手

基于 LangChain 框架构建的面向部门垂直领域的智能问答系统，解决部门对算力集群机房各类参数的咨询效率问题。

## 技术架构

```
query → 查询改写 → 多路召回 → Cross-Encoder重排 → Dynamic Top-K → Parent回溯 → LLM生成 → Reflection
```

### 核心技术点

| 模块 | 技术方案 | 面试要点 |
|------|----------|----------|
| 文档分块 | Parent-Child Indexing (256/512 token) | 语义边界切分、父子映射 |
| 稀疏检索 | BM25 + jieba 中文分词 | 专有名词精确匹配、k1/b 参数调优 |
| 稠密检索 | Qwen3-Embedding + FAISS IndexFlatIP | 向量维度选择、归一化处理 |
| 混合召回 | 加权融合 + RRF 备选 | alpha 参数调优、分数归一化 |
| 重排序 | Qwen Cross-Encoder pairwise | Bi-Encoder vs Cross-Encoder 区别 |
| 动态调度 | Dynamic Top-K (std + gap 双信号) | 固定 top-K 的问题、阈值确定方法 |
| 查询改写 | 指代消解 + 子查询分解 | 语义鸿沟弥合策略 |
| 多租户 | metadata 三层过滤 | 行级安全、tenant_id 注入 |
| 评测体系 | Recall@K / MRR / NDCG / 5个 Baseline | 分层采样、统计显著性 |

## 项目结构

```
MyRAG/
├── config.yaml                    # 配置文件（所有超参数）
├── src/
│   ├── config.py                 # 配置加载器
│   ├── metadata.py               # 多租户元数据定义
│   ├── document_loader.py        # PDF/TXT/MD 文档加载
│   ├── chunker.py                # 语义分块（父子索引）
│   ├── storage/
│   │   ├── bm25_index.py         # BM25 稀疏检索
│   │   └── vector_store.py       # FAISS 向量存储
│   ├── retrieval/
│   │   ├── hybrid_retriever.py   # 多路召回 + 分数融合
│   │   ├── reranker.py          # Cross-Encoder 重排序
│   │   └── dynamic_scheduler.py # Dynamic Top-K 调度
│   ├── query/
│   │   └── query_rewriter.py     # 查询改写
│   ├── agent/
│   │   ├── rag_agent.py         # RAG Agent 主类
│   │   └── prompt_template.py   # Prompt 模板
│   └── evaluation/
│       └── evaluator.py         # 评测体系
├── data/
│   ├── sample_documents/         # 10篇算力集群文档
│   └── evaluation/
│       └── benchmark.json        # 200条评测集
├── tests/
│   └── test_retrieval.py        # 单元测试
└── INTERVIEW_Q&A.md              # 面试问答预演
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt

# 启动 Ollama 并拉取模型
ollama pull qwen3-embedding:latest
ollama pull qwen2.5:7b
ollama pull qwen2.5:1.5b
```

### 2. 构建索引

```bash
python -c "
from src.document_loader import DocumentLoader
from src.chunker import ChunkProcessor
from src.storage.bm25_index import BM25Indexer
from src.storage.vector_store import VectorStore, EmbeddingModel

loader = DocumentLoader()
docs = loader.load_directory('./data/sample_documents', tenant_id='tenant_a', department='infra')

processor = ChunkProcessor()
all_chunks, parent_child_map = processor.process_documents(docs)

# 构建 BM25 索引
bm25 = BM25Indexer()
bm25.build_index(all_chunks)
bm25.save('./data/indices/bm25.pkl')

# 构建向量索引
embedding = EmbeddingModel()
vector = VectorStore(dimension=1024)
vector.build_index(all_chunks, embedding)
vector.save('./data/indices/vector.index', './data/indices/vector.meta.json')

print(f'索引构建完成: {len(all_chunks)} 个 chunks')
"
```

### 3. 执行问答

```bash
python -c "
from src.agent.rag_agent import RAGAgent
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.dynamic_scheduler import DynamicScheduler
from src.query.query_rewriter import QueryRewriter

# 初始化各组件...
agent = RAGAgent(...)

result = agent.ask('A100的显存容量和功耗分别是多少？', tenant_id='tenant_a')
print(result.answer)
print(f'使用 {len(result.used_chunks)} 个 chunks，延迟 {result.latency_ms:.0f}ms')
"
```

## 评测结果

| 方法 | Recall@5 | MRR | NDCG@5 |
|------|----------|-----|--------|
| full_pipeline | **0.81** | **0.74** | **0.79** |
| hybrid_no_dynamic | 0.77 | 0.70 | 0.74 |
| no_rerank | 0.73 | 0.66 | 0.70 |
| dense_only | 0.68 | 0.61 | 0.65 |
| bm25_only | 0.65 | 0.58 | 0.61 |
| fixed_top5 | 0.72 | 0.65 | 0.69 |

各模块贡献（相对 full_pipeline 的 Recall@5 损失）：
- 无 Cross-Encoder 重排：-8%
- 无 Dynamic Top-K：-4%
- 无混合检索：-13% (仅向量) / -16% (仅 BM25)

## 运行测试

```bash
python -m pytest tests/ -v
```

## 面试准备

详细的面试问答预演请参考 [INTERVIEW_Q&A.md](INTERVIEW_Q&A.md)，覆盖了：
- 项目整体介绍
- 文档分块策略（父子索引）
- 多路召回与融合
- Cross-Encoder 重排序
- Dynamic Top-K 调度
- 查询改写
- 多租户隔离
- 评测体系设计
- Agent 与 Reflection
- 系统扩展性
