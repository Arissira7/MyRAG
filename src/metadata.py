"""多租户元数据定义与管理。

定义 Chunk 元数据结构，提供租户隔离的工具函数。
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class ChunkMetadata:
    """Chunk 元数据。

    每个 chunk 都携带完整的 metadata，用于多租户隔离和检索过滤。
    """

    tenant_id: str
    department: str
    doc_id: str
    doc_title: str
    chunk_id: str
    parent_id: Optional[str] = None
    doc_type: Optional[str] = None
    author: Optional[str] = None
    created_at: Optional[str] = None
    level: Optional[str] = None
    section_title: Optional[str] = None
    chunk_index: int = 0
    total_chunks: int = 1
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典。"""
        return {
            "tenant_id": self.tenant_id,
            "department": self.department,
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "chunk_id": self.chunk_id,
            "parent_id": self.parent_id,
            "doc_type": self.doc_type,
            "author": self.author,
            "created_at": self.created_at,
            "level": self.level,
            "section_title": self.section_title,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            **self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChunkMetadata":
        """从字典创建。"""
        known_fields = {
            "tenant_id", "department", "doc_id", "doc_title", "chunk_id",
            "parent_id", "doc_type", "author", "created_at", "level",
            "section_title", "chunk_index", "total_chunks",
        }
        extra = {k: v for k, v in data.items() if k not in known_fields}
        return cls(
            tenant_id=data["tenant_id"],
            department=data["department"],
            doc_id=data["doc_id"],
            doc_title=data["doc_title"],
            chunk_id=data["chunk_id"],
            parent_id=data.get("parent_id"),
            doc_type=data.get("doc_type"),
            author=data.get("author"),
            created_at=data.get("created_at"),
            level=data.get("level"),
            section_title=data.get("section_title"),
            chunk_index=data.get("chunk_index", 0),
            total_chunks=data.get("total_chunks", 1),
            extra=extra,
        )


@dataclass
class RetrievedChunk:
    """检索返回的 Chunk 结构。"""

    chunk_id: str
    text: str
    metadata: ChunkMetadata
    score: float = 0.0
    source: str = "hybrid"  # "bm25", "dense", or "hybrid"
    rank: int = 0


def filter_chunks_by_tenant(
    chunks: list[RetrievedChunk], tenant_id: str
) -> list[RetrievedChunk]:
    """根据 tenant_id 过滤 chunks，实现租户隔离。

    这是多租户安全的第一道防线——确保检索结果只包含当前租户的数据。
    """
    return [c for c in chunks if c.metadata.tenant_id == tenant_id]


def build_tenant_filter(metadata: ChunkMetadata) -> dict[str, Any]:
    """构建用于向量检索的租户过滤条件。

    FAISS 和大多数向量数据库支持在检索时传入 filter 参数，
    这样过滤在向量搜索阶段就完成，性能最优。
    """
    return {
        "tenant_id": metadata.tenant_id,
        "department": metadata.department,
    }
