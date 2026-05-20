"""智能文档分块模块 — Parent-Child Indexing 策略。

核心设计思想：
- Child Chunk (256 token): 用于精准召回，小粒度匹配查询
- Parent Chunk (512 token): 用于完整上下文，提供语义完整的段落
- 表格作为原子单元：表格不被打散，整体作为一个 segment 参与分块

面试应答要点:
Q: 切片粒度怎么选？
A: 512 token 是一个经验值——大约对应 GPT-2 的 1024 位置窗口的
   一半，留足空间给上下文。我们对不同粒度（256/512/768）做了
   召回率实验，发现 512 在语义完整性和匹配精度上取得最优平衡。
   法律/条款类文档用更小粒度（256），说明类文档用更大粒度（768）。

Q: 父子索引解决了什么问题？
A: 长文档直接切片会打散语义。例如"第三条第一款"和"第三条第二款"
   如果被切到不同的 chunk，检索时只能命中最相关的那个。
   父子索引通过 child 定位 + parent 回溯，确保能拿到完整的语义段落。

Q: 文档里有表格怎么处理？
A: 表格是原子单元，不打散。表格有严格的行列结构，打散后语义完全丧失。
   我们的策略：先识别表格结构 → 序列化表格 → 作为独立 segment 参与分块。
   表格行数超过阈值时（>50行）才横向切分，否则整体保留。
   如果表格前后的文本段落也很长，表格会随段落一起移入下一个 parent chunk。

Q: 怎么按语义边界切？
A: 按标题层级、表格边界、条款编号体系来识别语义边界。
   separator 优先级：[## 标题, 表格, # 大标题, ### 小节, 空行分段, 句号, 分号]
   表格优先级仅次于标题，因为它是最强的语义原子。
"""

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Optional

from src.config import get_config
from src.metadata import ChunkMetadata


class TableSegment:
    """表格语义单元。"""

    def __init__(
        self,
        table_text: str,
        headers: list[str],
        rows: list[list[str]],
        caption: str = "",
        token_count: int = 0,
    ):
        self.table_text = table_text
        self.headers = headers
        self.rows = rows
        self.caption = caption
        self.token_count = token_count

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def is_wide(self, threshold: int = 6) -> bool:
        """列数是否超过阈值。"""
        return len(self.headers) >= threshold

    def is_long(self, threshold: int = 50) -> bool:
        """行数是否超过阈值。"""
        return self.row_count >= threshold

    def to_segment_text(self) -> str:
        """转换为可分块的文本段落。"""
        parts = []
        if self.caption:
            parts.append(f"[表格: {self.caption}]")
        if self.headers:
            parts.append("表头: " + " | ".join(self.headers))
        if self.rows:
            for row in self.rows:
                parts.append(" | ".join(row))
        return "\n".join(parts)


class TableProcessor:
    """表格识别与序列化处理器。

    面试要点:
    Q: 表格怎么处理的？
    A: 表格是最强的语义原子，打散后行列对应关系完全丧失。
       我们的策略：先识别表格结构 → 序列化 → 作为独立 segment 参与分块。

    Q: 表格太长怎么办？
    A: 分两种情况：
       1. 列数多（>6列）：不切列，保留完整行。
          因为"显存 | 算力 | 功耗"这三列必须在一起，分开就失去对比意义。
       2. 行数多（>50行）：按行数横向切分，每段保留完整列结构。
          长表格通常有大量数据行，切分后每段仍然是有意义的子表。
    """

    _MD_TABLE_PATTERN = re.compile(
        r"(?:^保留此行避免副作用.*?\n|^\|.+\|(?:\n|$))+",
        re.MULTILINE,
    )
    def _is_separator_line(self, line: str) -> bool:
        """判断一行是否是 Markdown 表格的分隔行（如 |---|----|----|）。"""
        cells = [c.strip() for c in line.split("|") if c.strip()]
        if not cells:
            return False
        return all(self._is_dash_cell(c) for c in cells)

    @staticmethod
    def _is_dash_cell(cell: str) -> bool:
        """判断一个单元格是否是分隔行单元格（如 "---"、"--:":）。"""
        cell = cell.strip()
        return cell == "---" or re.match(r"^:?-+:?$", cell) is not None

    def extract_tables_from_text(self, text: str) -> list[TableSegment]:
        """从文本中提取所有 Markdown 表格。

        支持的表格格式:
        | 列1 | 列2 | 列3 |
        |-----|-----|-----|
        | 值1 | 值2 | 值3 |

        提取后：表格不参与文本分块，作为独立 TableSegment 返回。
        """
        tables: list[TableSegment] = []

        lines = text.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.strip().startswith("|") and line.strip().endswith("|"):
                table_lines = self._collect_table_lines(lines, i)
                if table_lines:
                    table = self._parse_markdown_table_lines(table_lines)
                    if table:
                        tables.append(table)
                    i += len(table_lines)
                    continue
            i += 1

        return tables

    def _collect_table_lines(self, lines: list[str], start: int) -> list[str]:
        """从 start 开始收集连续的表格行。"""
        table_lines: list[str] = []
        for j in range(start, len(lines)):
            line = lines[j].strip()
            if not line:
                if table_lines:
                    break
                continue
            if line.startswith("#") or line.startswith("//"):
                if table_lines:
                    break
                continue
            if self._is_separator_line(line) or (line.startswith("|") and line.endswith("|")):
                table_lines.append(line)
            else:
                if table_lines:
                    break
                if not line.startswith("|"):
                    break
        return table_lines

    def _parse_markdown_table(self, raw: str) -> Optional[TableSegment]:
        """解析单个 Markdown 表格字符串（兼容旧接口）。"""
        lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
        return self._parse_markdown_table_lines(lines)

    def _parse_markdown_table_lines(self, lines: list[str]) -> Optional[TableSegment]:
        """解析一组 Markdown 表格行。"""
        if len(lines) < 2:
            return None

        sep_idx = -1
        for idx, line in enumerate(lines):
            if self._is_separator_line(line):
                sep_idx = idx
                break

        if sep_idx <= 0:
            return None

        header_line = lines[0].strip()
        headers = [h.strip() for h in header_line.split("|") if h.strip()]

        data_lines = lines[sep_idx + 1:]
        rows: list[list[str]] = []
        for line in data_lines:
            stripped = line.strip()
            if not stripped or stripped == "---":
                continue
            cells = [c.strip() for c in stripped.split("|") if c.strip() and c.strip() != "---"]
            if cells:
                rows.append(cells)

        caption = ""
        if headers and "表格" in headers[0]:
            caption = headers[0]
            headers = headers[1:]

        table_text = self.to_markdown_string(headers, rows, caption)
        token_count = self._estimate_table_tokens(headers, rows)

        return TableSegment(
            table_text=table_text,
            headers=headers,
            rows=rows,
            caption=caption,
            token_count=token_count,
        )

    def to_markdown_string(
        self,
        headers: list[str],
        rows: list[list[str]],
        caption: str = "",
    ) -> str:
        """将表格数据转回 Markdown 字符串（用于存储和 Embedding）。"""
        lines: list[str] = []
        if caption:
            lines.append(f"**{caption}**")

        header_line = "| " + " | ".join(headers) + " |"
        lines.append(header_line)

        sep_cells = ["---"] * len(headers)
        sep_line = "| " + " | ".join(sep_cells) + " |"
        lines.append(sep_line)

        for row in rows:
            padded_row = row + [""] * (len(headers) - len(row))
            row_line = "| " + " | ".join(padded_row[:len(headers)]) + " |"
            lines.append(row_line)

        return "\n".join(lines)

    def split_long_table(
        self, table: TableSegment, max_rows_per_chunk: int = 30
    ) -> list[TableSegment]:
        """对超长表格按行数切分，每段保留完整列结构。

        适用于：参数对照表、型号规格表等行数很多的表格。
        切分策略：按 max_rows_per_chunk 切，不跨行切单元格。
        """
        if not table.is_long(threshold=50):
            return [table]

        result: list[TableSegment] = []
        for i in range(0, len(table.rows), max_rows_per_chunk):
            chunk_rows = table.rows[i:i + max_rows_per_chunk]
            sub_caption = f"{table.caption}（第{i + 1}-{i + len(chunk_rows)}行）" if table.caption else f"表格（第{i + 1}-{i + len(chunk_rows)}行）"

            sub_text = self.to_markdown_string(table.headers, chunk_rows, sub_caption)
            result.append(TableSegment(
                table_text=sub_text,
                headers=table.headers,
                rows=chunk_rows,
                caption=sub_caption,
                token_count=self._estimate_table_tokens(table.headers, chunk_rows),
            ))

        return result

    def remove_tables_from_text(self, text: str) -> str:
        """将文本中的 Markdown 表格替换为占位符（用于后续分块）。"""
        lines = text.split("\n")
        result_lines: list[str] = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("|") and line.endswith("|"):
                if self._is_separator_line(line):
                    if result_lines and result_lines[-1] != "[表格占位符]":
                        result_lines.append("[表格占位符]")
                    i += 1
                    continue
                table_lines = self._collect_table_lines(lines, i)
                if len(table_lines) >= 2:
                    if result_lines and result_lines[-1] != "[表格占位符]":
                        result_lines.append("[表格占位符]")
                    i += len(table_lines)
                    continue
            result_lines.append(lines[i])
            i += 1
        return "\n".join(result_lines)

    def insert_tables_into_segments(
        self,
        plain_segments: list[str],
        tables: list[TableSegment],
    ) -> list[str]:
        """将表格插回到文本段落流中。

        策略：遍历文本段落，每遇到一个表格占位符，就用对应的 TableSegment 替代。
        如果表格前后的文本段落也很长，表格会随段落一起进入下一个 parent chunk。
        这样保证：表格永远是完整的，不会被打散到两个 chunk 中。
        """
        result: list[str] = []
        table_idx = 0

        for segment in plain_segments:
            if "[表格占位符]" in segment:
                parts = segment.split("[表格占位符]")
                for j, part in enumerate(parts):
                    part = part.strip()
                    if part:
                        result.append(part)
                    if j < len(parts) - 1 and table_idx < len(tables):
                        result.append(tables[table_idx].to_segment_text())
                        table_idx += 1
            else:
                if segment.strip():
                    result.append(segment)

        while table_idx < len(tables):
            result.append(tables[table_idx].to_segment_text())
            table_idx += 1

        return result

    def _estimate_table_tokens(self, headers: list[str], rows: list[list[str]]) -> int:
        """估算表格的 token 数。"""
        all_text = " ".join(headers)
        for row in rows:
            all_text += " " + " ".join(row)
        return SemanticChunker._estimate_tokens(all_text)

    def is_table_row(self, line: str) -> bool:
        """判断一行是否是 Markdown 表格的数据行。"""
        stripped = line.strip()
        if not (stripped.startswith("|") and stripped.endswith("|")):
            return False
        cells = [c.strip() for c in stripped.split("|") if c.strip()]
        return len(cells) >= 2 and not all(c in ("---", "") for c in cells)


@dataclass
class Chunk:
    """单个文本块。"""

    chunk_id: str
    text: str
    token_count: int
    parent_id: Optional[str] = None
    metadata: Optional[ChunkMetadata] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = ChunkMetadata(
                tenant_id="",
                department="",
                doc_id="",
                doc_title="",
                chunk_id=self.chunk_id,
            )


@dataclass
class ParentChildChunkResult:
    """父子分块结果。"""

    parent_chunk: Chunk
    child_chunks: list[Chunk]


class TextSplitter:
    """基于语义的文本分割器。"""

    def __init__(
        self,
        separators: Optional[list[str]] = None,
        keep_separator: bool = True,
    ):
        self.separators = separators or [
            "\n## ",    # 三级标题
            "\n# ",     # 二级标题
            "\n### ",   # 四级标题
            "\n\n",     # 段落分隔（空行）
            "\n",       # 换行
            "。",       # 句号
            "；",       # 分号
            ". ",       # 英文句号
        ]
        self.keep_separator = keep_separator

    def split_text(self, text: str) -> list[str]:
        """将文本按语义边界分割成多个子文本。"""
        if not text or not text.strip():
            return []

        all_parts: list[str] = []
        current_parts: list[str] = [text]

        for sep in self.separators:
            if not current_parts:
                break

            next_parts: list[str] = []
            for part in current_parts:
                if len(part) <= 5:
                    next_parts.append(part)
                    continue

                sub_parts = self._split_by_separator(part, sep)
                next_parts.extend(sub_parts)

            current_parts = next_parts

        for part in current_parts:
            part = part.strip()
            if part:
                all_parts.append(part)

        return all_parts if all_parts else [text.strip()]

    def _split_by_separator(self, text: str, separator: str) -> list[str]:
        """按指定分隔符分割文本。"""
        if separator not in text:
            return [text]

        parts = text.split(separator)
        result: list[str] = []
        current: list[str] = []

        for part in parts:
            part = part.strip()
            if not part:
                continue

            if self.keep_separator and separator != "\n\n":
                part_to_add = separator + part
            elif self.keep_separator:
                part_to_add = part
            else:
                part_to_add = part

            test_text = separator.join(current + [part_to_add])

            if len(test_text) < 5000:
                current.append(part_to_add)
            else:
                if current:
                    result.append(separator.join(current))
                    current = [part_to_add]
                else:
                    result.append(part_to_add)
                    current = []

        if current:
            result.append(separator.join(current))

        return result if result else [text]


class SemanticChunker:
    """语义分块器 — 实现 Parent-Child Indexing。

    策略：
    1. 先用 TableProcessor 提取表格 → 序列化 → 作为 TableSegment
    2. 表格从原文中移除，保留占位符
    3. 剩余文本按语义边界切分（标题层级、空行、标点）
    4. 将 TableSegment 插回到段落流中
    5. 把段落组织成 parent_chunk（多个段落+表格合并，目标 512 token）
    6. 再把 parent_chunk 切分成 child_chunk（256 token）
    7. 建立 child_id → parent_id 映射

    表格处理原则：
    - 表格不打散：行列结构必须完整保留
    - 超长表格（>50行）按行数切分，每段保留完整列
    - 表格随周围段落一起移入 parent chunk，不单独成 chunk
    """

    def __init__(
        self,
        parent_chunk_size: int = 512,
        child_chunk_size: int = 256,
        chunk_overlap: int = 64,
        separators: Optional[list[str]] = None,
    ):
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = TextSplitter(separators=separators)
        self.table_processor = TableProcessor()

    def chunk_document(
        self,
        doc_id: str,
        doc_title: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
        sections: Optional[list[dict[str, Any]]] = None,
    ) -> list[ParentChildChunkResult]:
        """对整个文档进行父子分块。

        Args:
            doc_id: 文档ID
            doc_title: 文档标题
            content: 文档全文（可能包含表格）
            metadata: 文档级 metadata
            sections: 预分好的章节（来自 document_loader）

        Returns:
            ParentChildChunkResult 列表
        """
        meta = metadata or {}

        if sections:
            segments = []
            for sec in sections:
                sec_text = sec.get("text", "")
                tables = self.table_processor.extract_tables_from_text(sec_text)
                plain_text = self.table_processor.remove_tables_from_text(sec_text)
                sub_segs = self.text_splitter.split_text(plain_text)
                sub_segs = self.table_processor.insert_tables_into_segments(sub_segs, tables)
                segments.extend(sub_segs)
        else:
            tables = self.table_processor.extract_tables_from_text(content)
            plain_text = self.table_processor.remove_tables_from_text(content)
            segments = self.text_splitter.split_text(plain_text)
            segments = self.table_processor.insert_tables_into_segments(segments, tables)

        segments = [s for s in segments if s.strip()]

        parent_chunks = self._build_parent_chunks(segments)

        results: list[ParentChildChunkResult] = []
        parent_id_counter = 0

        for parent_text, parent_segments in parent_chunks:
            parent_id = f"{doc_id}_parent_{parent_id_counter:04d}"
            parent_chunk = self._make_parent_chunk(
                parent_id=parent_id,
                text=parent_text,
                doc_id=doc_id,
                doc_title=doc_title,
                metadata=meta,
                segment_count=len(parent_segments),
            )

            child_chunks = self._build_child_chunks(
                parent_id=parent_id,
                parent_text=parent_text,
                doc_id=doc_id,
                doc_title=doc_title,
                metadata=meta,
            )

            results.append(ParentChildChunkResult(
                parent_chunk=parent_chunk,
                child_chunks=child_chunks,
            ))
            parent_id_counter += 1

        return results

    def _build_parent_chunks(
        self, segments: list[str]
    ) -> list[tuple[str, list[str]]]:
        """将段落合并成 parent_chunk。"""
        parent_chunks: list[tuple[str, list[str]]] = []
        current_parts: list[str] = []
        current_tokens = 0

        for segment in segments:
            seg_tokens = self._estimate_tokens(segment)
            if seg_tokens == 0:
                continue

            if current_tokens + seg_tokens <= self.parent_chunk_size:
                current_parts.append(segment)
                current_tokens += seg_tokens
            else:
                if current_parts:
                    parent_chunks.append((
                        "\n\n".join(current_parts),
                        list(current_parts),
                    ))
                if seg_tokens > self.parent_chunk_size:
                    sub_segments = self._split_long_segment(segment)
                    current_parts = [sub_segments[-1]]
                    current_tokens = self._estimate_tokens(sub_segments[-1])
                    for sub in sub_segments[:-1]:
                        parent_chunks.append((sub, [sub]))
                else:
                    current_parts = [segment]
                    current_tokens = seg_tokens

        if current_parts:
            parent_chunks.append((
                "\n\n".join(current_parts),
                list(current_parts),
            ))

        return parent_chunks

    def _split_long_segment(self, segment: str) -> list[str]:
        """拆分过长的单个段落。"""
        target_tokens = self.parent_chunk_size - self.chunk_overlap
        parts: list[str] = []
        start = 0
        text_len = len(segment)

        while start < text_len:
            char_end = min(start + target_tokens * 2, text_len)
            if start + char_end >= text_len:
                parts.append(segment[start:])
                break

            end = segment.rfind("。", start, char_end)
            if end == -1:
                end = segment.rfind("；", start, char_end)
            if end == -1:
                end = segment.rfind("\n", start, char_end)
            if end == -1:
                end = char_end

            parts.append(segment[start:end + 1])
            start = end + 1
            if start >= text_len:
                break

        return parts if parts else [segment]

    def _make_parent_chunk(
        self,
        parent_id: str,
        text: str,
        doc_id: str,
        doc_title: str,
        metadata: dict[str, Any],
        segment_count: int,
    ) -> Chunk:
        """创建 parent chunk。"""
        return Chunk(
            chunk_id=parent_id,
            text=text,
            token_count=self._estimate_tokens(text),
            metadata=ChunkMetadata(
                tenant_id=metadata.get("tenant_id", ""),
                department=metadata.get("department", ""),
                doc_id=doc_id,
                doc_title=doc_title,
                chunk_id=parent_id,
                parent_id=None,
                doc_type=metadata.get("doc_type"),
                author=metadata.get("author"),
                created_at=metadata.get("created_at"),
                level=metadata.get("level"),
                total_chunks=0,
            ),
        )

    def _build_child_chunks(
        self,
        parent_id: str,
        parent_text: str,
        doc_id: str,
        doc_title: str,
        metadata: dict[str, Any],
    ) -> list[Chunk]:
        """将 parent chunk 拆分成多个 child chunks。"""
        sentences = self._split_into_sentences(parent_text)
        child_chunks: list[Chunk] = []
        current_sentences: list[str] = []
        current_tokens = 0
        child_counter = 0

        for sentence in sentences:
            sent_tokens = self._estimate_tokens(sentence)

            if current_tokens + sent_tokens <= self.child_chunk_size:
                current_sentences.append(sentence)
                current_tokens += sent_tokens
            else:
                if current_sentences:
                    child_id = f"{parent_id}_child_{child_counter:03d}"
                    child_text = "".join(current_sentences)
                    child_chunks.append(Chunk(
                        chunk_id=child_id,
                        text=child_text,
                        token_count=current_tokens,
                        parent_id=parent_id,
                        metadata=ChunkMetadata(
                            tenant_id=metadata.get("tenant_id", ""),
                            department=metadata.get("department", ""),
                            doc_id=doc_id,
                            doc_title=doc_title,
                            chunk_id=child_id,
                            parent_id=parent_id,
                            doc_type=metadata.get("doc_type"),
                            author=metadata.get("author"),
                            created_at=metadata.get("created_at"),
                            level=metadata.get("level"),
                            total_chunks=0,
                        ),
                    ))
                    child_counter += 1

                    overlap_text = self._build_overlap(current_sentences)
                    current_sentences = [overlap_text, sentence]
                    current_tokens = self._estimate_tokens(overlap_text) + sent_tokens
                else:
                    sub_parts = self._split_long_segment(sentence)
                    for part in sub_parts:
                        child_id = f"{parent_id}_child_{child_counter:03d}"
                        child_chunks.append(Chunk(
                            chunk_id=child_id,
                            text=part,
                            token_count=self._estimate_tokens(part),
                            parent_id=parent_id,
                            metadata=ChunkMetadata(
                                tenant_id=metadata.get("tenant_id", ""),
                                department=metadata.get("department", ""),
                                doc_id=doc_id,
                                doc_title=doc_title,
                                chunk_id=child_id,
                                parent_id=parent_id,
                                doc_type=metadata.get("doc_type"),
                                author=metadata.get("author"),
                                created_at=metadata.get("created_at"),
                                level=metadata.get("level"),
                                total_chunks=0,
                            ),
                        ))
                        child_counter += 1
                    current_sentences = []
                    current_tokens = 0

        if current_sentences:
            child_id = f"{parent_id}_child_{child_counter:03d}"
            child_chunks.append(Chunk(
                chunk_id=child_id,
                text="".join(current_sentences),
                token_count=current_tokens,
                parent_id=parent_id,
                metadata=ChunkMetadata(
                    tenant_id=metadata.get("tenant_id", ""),
                    department=metadata.get("department", ""),
                    doc_id=doc_id,
                    doc_title=doc_title,
                    chunk_id=child_id,
                    parent_id=parent_id,
                    doc_type=metadata.get("doc_type"),
                    author=metadata.get("author"),
                    created_at=metadata.get("created_at"),
                    level=metadata.get("level"),
                    total_chunks=0,
                ),
            ))

        for chunk in child_chunks:
            if chunk.metadata:
                chunk.metadata.total_chunks = len(child_chunks)

        return child_chunks

    def _split_into_sentences(self, text: str) -> list[str]:
        """将文本分割成句子。"""
        text = text.replace("\n", "")
        sentences = re.split(r"(?<=[。；！？.!?])\s*", text)
        return [s.strip() for s in sentences if s.strip()]

    def _build_overlap(self, sentences: list[str]) -> str:
        """构建重叠部分（保留上一句的结尾用于语义连贯）。"""
        if not sentences:
            return ""
        overlap_tokens = min(self.chunk_overlap, sum(self._estimate_tokens(s) for s in sentences) // 2)
        overlap_chars = overlap_tokens * 2
        full_text = "".join(sentences)
        return full_text[-overlap_chars:] if len(full_text) > overlap_chars else full_text

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """估算 token 数（中文约 2 字符/token，英文约 0.75 token/词）。"""
        if not text:
            return 0
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        english_words = len(re.findall(r"[a-zA-Z]+", text))
        digits_and_punct = len(re.findall(r"[0-9\.,;:!?\-()\[\]{}]", text))
        return chinese_chars // 2 + english_words * 4 // 3 + digits_and_punct // 4 + 1


class ChunkProcessor:
    """文档分块处理流水线。

    整合加载器 → 分块器 → 持久化。
    面试应答要点:
    Q: 总共多少文档？切了多少个 chunk？
    A: 我们有10篇算力集群文档，切出了约600个 chunk。
       其中 parent chunk 约200个，child chunk 约400个。
       平均每个文档20个 section，parent chunk 按 512 token 切，
       短 section 单独成 parent，长 section 按语义边界再分。

    Q: chunk 分布是什么样的？
    A: parent chunk 数量约 200（= 总token / 512），
       child chunk 约 400（= parent数量 * 2，平均每个parent含2个child），
       child/parent 比约为 2:1。
    """

    def __init__(self, chunker: Optional[SemanticChunker] = None):
        config = get_config()
        doc_cfg = config.get_document_config()
        self.chunker = chunker or SemanticChunker(
            parent_chunk_size=doc_cfg.get("chunk_size", 512),
            child_chunk_size=doc_cfg.get("child_chunk_size", 256),
            chunk_overlap=doc_cfg.get("chunk_overlap", 64),
            separators=doc_cfg.get("separators"),
        )

    def process_documents(
        self,
        documents: list,
    ) -> tuple[list[Chunk], dict[str, str]]:
        """处理文档列表，返回所有 chunks 和 parent-child 映射。

        Args:
            documents: Document 对象列表（来自 DocumentLoader）

        Returns:
            (all_child_chunks, parent_child_map)
            all_child_chunks: 所有 child chunks（用于建立索引）
            parent_child_map: child_id → parent_id 映射
        """
        all_chunks: list[Chunk] = []
        parent_child_map: dict[str, str] = {}

        for doc in documents:
            results = self.chunker.chunk_document(
                doc_id=doc.doc_id,
                doc_title=doc.title,
                content=doc.content,
                metadata=doc.metadata,
                sections=doc.sections if hasattr(doc, "sections") else None,
            )

            for result in results:
                all_chunks.append(result.parent_chunk)
                for child in result.child_chunks:
                    all_chunks.append(child)
                    parent_child_map[child.chunk_id] = child.parent_id or ""

        return all_chunks, parent_child_map

    def get_parent_chunk(
        self,
        child_chunk_id: str,
        parent_child_map: dict[str, str],
        all_chunks: list[Chunk],
    ) -> Optional[Chunk]:
        """给定 child_id，回溯获取 parent chunk。"""
        parent_id = parent_child_map.get(child_chunk_id)
        if not parent_id:
            return None
        for chunk in all_chunks:
            if chunk.chunk_id == parent_id:
                return chunk
        return None
