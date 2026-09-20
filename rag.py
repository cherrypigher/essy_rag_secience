"""极简科研 RAG 命令行工具。

读取 ``papers/`` 中的 PDF，按固定字符数切块并生成本地 Embedding，
持久化到 ChromaDB；提问时检索最相关的 3 个片段，交给本机 Ollama 的
``deepseek-r1:7b`` 生成回答，并列出实际检索到的论文文件名。
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


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


def build_index(config: Config) -> None:
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
