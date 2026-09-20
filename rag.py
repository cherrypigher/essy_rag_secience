"""极简科研 RAG 命令行工具。

读取 ``papers/`` 中的 PDF，按固定字符数切块并生成本地 Embedding，
持久化到 ChromaDB；提问时检索最相关的 3 个片段，交给本机 Ollama 的
``deepseek-r1:7b`` 生成回答，并列出实际检索到的论文文件名。
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import chromadb
import fitz
from sentence_transformers import SentenceTransformer


class RagError(Exception):
    """用户可以理解并处理的 RAG 错误。"""


@dataclass(frozen=True)
class Config:
    """集中管理的默认配置，允许同名环境变量覆盖。"""

    papers_dir: Path
    chroma_dir: Path
    collection_name: str
    embedding_model: str
    ollama_base_url: str
    ollama_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    embedding_batch_size: int
    chroma_write_batch_size: int
    ollama_connect_timeout: int
    ollama_read_timeout: int

    @classmethod
    def from_env(cls) -> "Config":
        """从环境变量读取配置，校验失败时抛出带变量名的 RagError。"""

        chunk_size = _env_int("CHUNK_SIZE", 500, minimum=1)
        chunk_overlap = _env_int("CHUNK_OVERLAP", 50, minimum=0)
        if chunk_overlap >= chunk_size:
            raise RagError(
                f"CHUNK_OVERLAP 的值为 {chunk_overlap}，必须小于 CHUNK_SIZE 的 {chunk_size}。"
            )

        base_url = _env_str("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")

        return cls(
            papers_dir=Path(_env_str("PAPERS_DIR", "papers")),
            chroma_dir=Path(_env_str("CHROMA_DIR", "chroma_db")),
            collection_name=_env_str("COLLECTION_NAME", "science_papers"),
            embedding_model=_env_str("EMBEDDING_MODEL", "intfloat/multilingual-e5-small"),
            ollama_base_url=base_url,
            ollama_model=_env_str("OLLAMA_MODEL", "deepseek-r1:7b"),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=_env_int("TOP_K", 3, minimum=1),
            embedding_batch_size=_env_int("EMBEDDING_BATCH_SIZE", 32, minimum=1),
            chroma_write_batch_size=_env_int("CHROMA_WRITE_BATCH_SIZE", 256, minimum=1),
            ollama_connect_timeout=_env_int("OLLAMA_CONNECT_TIMEOUT", 5, minimum=1),
            ollama_read_timeout=_env_int("OLLAMA_READ_TIMEOUT", 300, minimum=1),
        )


@dataclass(frozen=True)
class PageSpan:
    """某一页非空文本在规范化全文中的字符区间，``end`` 为开区间。"""

    page_number: int
    start: int
    end: int


@dataclass(frozen=True)
class ExtractedPaper:
    """一篇 PDF 提取出的规范化全文和页码映射。"""

    path: Path
    relative_path: str
    file_sha256: str
    text: str
    page_spans: tuple[PageSpan, ...]


@dataclass(frozen=True)
class ChunkRecord:
    """一个待写入 ChromaDB 的文本块。"""

    id: str
    text: str
    source: str
    source_path: str
    page_start: int
    page_end: int
    chunk_index: int
    char_start: int
    char_end: int
    file_sha256: str


@dataclass(frozen=True)
class SearchHit:
    """一次检索命中的文本块，``rank`` 从 1 开始，``distance`` 越小越相关。"""

    rank: int
    text: str
    source: str
    source_path: str
    page_start: int
    page_end: int
    chunk_index: int
    distance: float


def _env_str(name: str, default: str) -> str:
    """读取字符串环境变量，去除首尾空白，空值视为未设置。"""

    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip()


def _env_int(name: str, default: int, *, minimum: int) -> int:
    """读取正整数环境变量，非法或越界时抛出带变量名的 RagError。"""

    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        raise RagError(
            f"环境变量 {name} 的值 {raw.strip()!r} 不是整数，"
            f"请设置为大于或等于 {minimum} 的整数。"
        ) from None
    if value < minimum:
        raise RagError(f"环境变量 {name} 的值为 {value}，必须大于或等于 {minimum}。")
    return value


def build_parser() -> argparse.ArgumentParser:
    """构建带 index / ask 两个子命令的命令行解析器。"""

    parser = argparse.ArgumentParser(
        prog="rag.py",
        description="极简科研 RAG：本地 PDF 索引与论文问答。",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "index",
        help="读取 papers/ 中的 PDF，重建 ChromaDB 向量索引",
    )

    ask_parser = subparsers.add_parser(
        "ask",
        help="检索相关论文片段并生成回答",
    )
    ask_parser.add_argument(
        "question",
        nargs="?",
        help="要回答的问题；省略时程序会提示输入一次",
    )

    return parser


_HASH_BLOCK_SIZE = 1024 * 1024
_PAGE_SEPARATOR = "\n\n"
_INLINE_WHITESPACE_RE = re.compile(r"[ \t\f\v]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def sha256_file(path: Path) -> str:
    """以 1 MiB 为单位流式计算文件 SHA-256，返回 64 位小写十六进制。"""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(_HASH_BLOCK_SIZE), b""):
                digest.update(block)
    except OSError as exc:
        raise RagError(f"无法读取文件 {path}：{exc}") from exc
    return digest.hexdigest()


def discover_pdfs(papers_dir: Path) -> list[Path]:
    """递归查找 papers_dir 下的 PDF，按相对路径 casefold 排序保证结果稳定。"""

    if not papers_dir.exists():
        raise RagError(
            f"论文目录 {papers_dir} 不存在，请创建该目录并放入 PDF 文件，"
            "或通过 PAPERS_DIR 指定其他目录。"
        )
    if not papers_dir.is_dir():
        raise RagError(f"路径 {papers_dir} 不是目录，请通过 PAPERS_DIR 指向存放 PDF 的目录。")

    pdf_paths = [
        path
        for path in papers_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    ]
    if not pdf_paths:
        raise RagError(f"在 {papers_dir} 下没有找到 PDF 文件，请先放入论文。")

    pdf_paths.sort(
        key=lambda path: (
            path.relative_to(papers_dir).as_posix().casefold(),
            path.relative_to(papers_dir).as_posix(),
        )
    )
    return pdf_paths


def normalize_page_text(text: str) -> str:
    """规范化单页 PDF 文本：压缩行内空白，但保留段落换行边界。"""

    cleaned = text.replace("\x00", " ")
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

    lines = [_INLINE_WHITESPACE_RE.sub(" ", line).strip() for line in cleaned.split("\n")]
    cleaned = "\n".join(lines)
    cleaned = _BLANK_LINES_RE.sub(_PAGE_SEPARATOR, cleaned)
    return cleaned.strip()


def extract_pdf(path: Path, papers_dir: Path) -> ExtractedPaper:
    """逐页提取 PDF 文本，并记录每页在规范化全文中的字符区间。"""

    file_sha256 = sha256_file(path)
    relative_path = path.relative_to(papers_dir).as_posix()

    document = None
    try:
        document = fitz.open(path)
        if document.needs_pass:
            raise RagError(f"PDF {relative_path} 已加密，请先移除密码保护后重新索引。")

        parts: list[str] = []
        page_spans: list[PageSpan] = []
        cursor = 0

        for page_number in range(1, document.page_count + 1):
            page_text = normalize_page_text(document.load_page(page_number - 1).get_text("text", sort=True))
            if not page_text:
                continue
            if parts:
                parts.append(_PAGE_SEPARATOR)
                cursor += len(_PAGE_SEPARATOR)
            parts.append(page_text)
            start = cursor
            cursor += len(page_text)
            page_spans.append(PageSpan(page_number=page_number, start=start, end=cursor))

        text = "".join(parts)
        if not text:
            raise RagError(
                f"PDF {relative_path} 没有可提取的文本层，可能是扫描版论文；"
                "首版不支持 OCR，请提供带文本层的 PDF。"
            )
    except RagError:
        raise
    except Exception as exc:
        raise RagError(f"提取 PDF {relative_path} 文本失败：{exc}") from exc
    finally:
        if document is not None:
            document.close()

    return ExtractedPaper(
        path=path,
        relative_path=relative_path,
        file_sha256=file_sha256,
        text=text,
        page_spans=tuple(page_spans),
    )


def pages_for_range(
    page_spans: Sequence[PageSpan],
    start: int,
    end: int,
) -> tuple[int, int]:
    """返回字符区间 [start, end) 覆盖的起止页码，无命中时抛错而不是伪造页码。"""

    matched = [
        span.page_number for span in page_spans if span.start < end and span.end > start
    ]
    if not matched:
        raise RagError(f"字符区间 [{start}, {end}) 没有对应到任何页，无法确定页码范围。")
    return matched[0], matched[-1]


def chunk_paper(
    paper: ExtractedPaper,
    chunk_size: int,
    overlap: int,
) -> list[ChunkRecord]:
    """按固定字符滑窗切块：chunk_size=500、overlap=50 时步长为 450。

    与朴素的 ``range(0, len(text), step)`` 相比，这里额外跳过“不产生任何新内容”的
    尾部窗口：例如 499 字符文本在 500/50 下，第二个窗口 [450, 499) 完全被第一个
    窗口 [0, 499) 覆盖，保留它只会得到一段纯重复文本。因此短于 500 字符的非空
    文本只产生一块，而 1000 字符仍然严格得到起点 0/450/900 的三块，相邻完整块
    的重叠字符数仍恰好是 overlap。
    """

    if chunk_size <= 0:
        raise RagError(f"chunk_size 必须大于 0，当前为 {chunk_size}。")
    if not 0 <= overlap < chunk_size:
        raise RagError(
            f"overlap 必须满足 0 <= overlap < chunk_size（当前 chunk_size 为 {chunk_size}），"
            f"当前 overlap 为 {overlap}。"
        )

    step = chunk_size - overlap
    chunks: list[ChunkRecord] = []
    covered_end: int | None = None

    for start in range(0, len(paper.text), step):
        end = min(start + chunk_size, len(paper.text))
        chunk_text = paper.text[start:end]
        if not chunk_text.strip():
            continue
        if covered_end is not None and end <= covered_end:
            continue

        page_start, page_end = pages_for_range(paper.page_spans, start, end)
        chunk_index = len(chunks)
        raw_id = (
            f"{paper.relative_path}\0{paper.file_sha256}\0"
            f"{chunk_index}\0{start}\0{end}"
        )
        chunks.append(
            ChunkRecord(
                id=hashlib.sha256(raw_id.encode("utf-8")).hexdigest(),
                text=chunk_text,
                source=paper.path.name,
                source_path=paper.relative_path,
                page_start=page_start,
                page_end=page_end,
                chunk_index=chunk_index,
                char_start=start,
                char_end=end,
                file_sha256=paper.file_sha256,
            )
        )
        covered_end = end

    if paper.text.strip() and not chunks:
        raise RagError(
            f"论文 {paper.relative_path} 有文本但没有产生任何文本块，请检查切块参数。"
        )
    return chunks


def load_embedding_model(model_name: str) -> SentenceTransformer:
    """加载 Embedding 模型；首次运行需要联网下载，失败时给出可操作提示。"""

    if not model_name.strip():
        raise RagError("EMBEDDING_MODEL 不能为空，请设置 Embedding 模型名称。")
    try:
        return SentenceTransformer(model_name)
    except Exception as exc:
        raise RagError(
            f"加载 Embedding 模型 {model_name} 失败：{exc}。"
            "首次运行需要联网下载模型，请检查网络连接后重试；"
            "也可以设置 EMBEDDING_MODEL 使用本地已缓存的模型。"
        ) from exc


def embed_passages(
    model: SentenceTransformer,
    chunks: Sequence[ChunkRecord],
    batch_size: int,
) -> list[list[float]]:
    """批量生成归一化的文档向量，输入文本统一添加 E5 的 passage: 前缀。"""

    if not chunks:
        raise RagError("没有可嵌入的文本块。")
    if batch_size <= 0:
        raise RagError(f"EMBEDDING_BATCH_SIZE 必须大于 0，当前为 {batch_size}。")

    texts = [f"passage: {chunk.text}" for chunk in chunks]
    try:
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
    except Exception as exc:
        raise RagError(f"生成文档向量失败：{exc}") from exc

    if hasattr(vectors, "tolist"):
        vectors = vectors.tolist()
    else:
        vectors = [list(vector) for vector in vectors]

    if len(vectors) != len(chunks):
        raise RagError(f"向量数量 {len(vectors)} 与文本块数量 {len(chunks)} 不一致。")
    if any(not vector for vector in vectors):
        raise RagError("Embedding 模型返回了空向量，请重试或更换模型。")
    return [list(vector) for vector in vectors]


def create_chroma_client(chroma_dir: Path) -> chromadb.PersistentClient:
    """创建本地持久化 ChromaDB 客户端，按需创建目录。"""

    try:
        chroma_dir.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=str(chroma_dir))
    except Exception as exc:
        raise RagError(f"初始化 ChromaDB 失败（路径 {chroma_dir}）：{exc}") from exc


def collection_names(client: Any) -> set[str]:
    """列出全部 collection 名称，兼容 list_collections 返回字符串或对象。"""

    names: set[str] = set()
    for item in client.list_collections():
        names.add(item if isinstance(item, str) else item.name)
    return names


def chunk_metadata(chunk: ChunkRecord) -> dict[str, str | int]:
    """生成写入 Chroma 的 metadata，只使用字符串和整数字段。"""

    return {
        "source": chunk.source,
        "source_path": chunk.source_path,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "chunk_index": chunk.chunk_index,
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
        "file_sha256": chunk.file_sha256,
    }


def build_index(config: Config) -> None:
    """全量重建索引：先完成全部提取和向量计算，再替换旧 collection。"""

    pdf_paths = discover_pdfs(config.papers_dir)

    chunks: list[ChunkRecord] = []
    succeeded = 0
    skipped = 0
    failures: list[str] = []

    for pdf_path in pdf_paths:
        relative_path = pdf_path.relative_to(config.papers_dir).as_posix()
        try:
            paper = extract_pdf(pdf_path, config.papers_dir)
            paper_chunks = chunk_paper(paper, config.chunk_size, config.chunk_overlap)
        except RagError as exc:
            skipped += 1
            failures.append(f"{relative_path}：{exc}")
            print(f"警告：跳过 {relative_path}：{exc}", file=sys.stderr)
            continue
        chunks.extend(paper_chunks)
        succeeded += 1

    if succeeded == 0:
        detail = "\n".join(f"- {failure}" for failure in failures)
        raise RagError(f"所有 PDF 都提取失败，索引未更新：\n{detail}")
    if not chunks:
        raise RagError("没有生成任何文本块，索引未更新。")

    model = load_embedding_model(config.embedding_model)
    embeddings = embed_passages(model, chunks, config.embedding_batch_size)

    client = create_chroma_client(config.chroma_dir)
    if config.collection_name in collection_names(client):
        client.delete_collection(config.collection_name)
    collection = client.create_collection(
        name=config.collection_name,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": config.embedding_model,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "schema_version": 1,
        },
    )

    for offset in range(0, len(chunks), config.chroma_write_batch_size):
        batch = chunks[offset : offset + config.chroma_write_batch_size]
        batch_embeddings = embeddings[offset : offset + config.chroma_write_batch_size]
        collection.upsert(
            ids=[chunk.id for chunk in batch],
            documents=[chunk.text for chunk in batch],
            embeddings=[list(vector) for vector in batch_embeddings],
            metadatas=[chunk_metadata(chunk) for chunk in batch],
        )

    stored = collection.count()
    if stored != len(chunks):
        raise RagError(f"写入校验失败：数据库中有 {stored} 条记录，预期 {len(chunks)} 条。")

    print("索引完成")
    print(f"- 成功论文：{succeeded}")
    print(f"- 跳过论文：{skipped}")
    print(f"- 文本块：{stored}")
    print(f"- ChromaDB：{config.chroma_dir}")
    print(f"- Collection：{config.collection_name}")


def answer_question(question: str, config: Config) -> None:
    """建立或重建 ChromaDB 索引（后续步骤实现）。"""

    raise RagError("索引功能尚未实现，请先完成后续实施步骤。")


def answer_question(question: str, config: Config) -> None:
    """回答问题并输出引用论文（后续步骤实现）。"""

    raise RagError("问答功能尚未实现，请先完成后续实施步骤。")


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口，返回进程退出码。"""

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = Config.from_env()
        if args.command == "index":
            build_index(config)
        else:
            question = args.question
            if question is None:
                question = input("请输入问题：")
            answer_question(question, config)
    except RagError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("已取消。", file=sys.stderr)
        return 130

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
