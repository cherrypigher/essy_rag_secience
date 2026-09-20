"""rag.py 的单元测试。

所有测试都使用临时目录和内存中的假对象：不访问网络、不下载真实 Embedding 模型、
不读取仓库 papers/ 下的真实论文。
"""

from __future__ import annotations

from pathlib import Path

import chromadb
import fitz
import pytest
import requests

import rag
from rag import (
    ChunkRecord,
    Config,
    ExtractedPaper,
    PageSpan,
    RagError,
    chunk_paper,
    discover_pdfs,
    extract_pdf,
    normalize_page_text,
    pages_for_range,
    sha256_file,
)

REPEATED_TEXT = "0123456789" * 100  # 1000 个可区分字符
SHA_PLACEHOLDER = "a" * 64
LONG_TEXT = ("The quick brown fox jumps over the lazy dog. " * 24).strip()  # 1079 字符


def make_paper(
    text: str,
    *,
    page_spans: tuple[PageSpan, ...] | None = None,
    file_sha256: str = SHA_PLACEHOLDER,
    relative_path: str = "论文1.pdf",
) -> ExtractedPaper:
    """构造用于切块测试的 ExtractedPaper，默认视为单页论文。"""

    if page_spans is None:
        page_spans = (PageSpan(page_number=1, start=0, end=len(text)),)
    return ExtractedPaper(
        path=Path("papers") / relative_path,
        relative_path=relative_path,
        file_sha256=file_sha256,
        text=text,
        page_spans=page_spans,
    )


def make_two_page_paper() -> ExtractedPaper:
    """构造一个 300 字符 + 分隔符 + 300 字符的两页论文。"""

    text = "A" * 300 + "\n\n" + "B" * 300
    page_spans = (
        PageSpan(page_number=1, start=0, end=300),
        PageSpan(page_number=2, start=302, end=602),
    )
    return make_paper(text, page_spans=page_spans)


def write_pdf(path: Path, pages: list[str]) -> None:
    """用 PyMuPDF 生成只含简单英文的 PDF，避免测试环境字体问题。

    使用 ``insert_textbox`` 让长文本在页面内换行；``insert_text`` 不会换行，
    超出页面宽度的文字不会被写入，导致长文本无法完整取回。
    """

    document = fitz.open()
    try:
        for page_text in pages:
            page = document.new_page(width=595, height=842)
            page.insert_textbox(
                fitz.Rect(50, 50, 545, 792),
                page_text,
                fontsize=11,
                fontname="helv",
            )
        document.save(path)
    finally:
        document.close()


class TestNormalizePageText:
    def test_removes_nul_and_unifies_newlines(self) -> None:
        raw = "第一行\x00内容\r\n第二行\r第三行"
        assert normalize_page_text(raw) == "第一行 内容\n第二行\n第三行"

    def test_compresses_inline_whitespace_but_keeps_paragraphs(self) -> None:
        raw = "a  \t b\x0c\x0bc\n\n\n\n\nd"
        assert normalize_page_text(raw) == "a b c\n\nd"

    def test_strips_surrounding_whitespace(self) -> None:
        assert normalize_page_text("  \n\n hello \n\n  ") == "hello"

    def test_empty_page_stays_empty(self) -> None:
        assert normalize_page_text("\x00\r\n   ") == ""


class TestChunkPaper:
    def test_thousand_characters_produce_three_chunks(self) -> None:
        chunks = chunk_paper(make_paper(REPEATED_TEXT), 500, 50)
        assert len(chunks) == 3
        assert [chunk.char_start for chunk in chunks] == [0, 450, 900]

    def test_chunk_ranges_are_exact(self) -> None:
        chunks = chunk_paper(make_paper(REPEATED_TEXT), 500, 50)
        assert [(c.char_start, c.char_end) for c in chunks] == [(0, 500), (450, 950), (900, 1000)]

    def test_adjacent_chunks_overlap_exactly_fifty_characters(self) -> None:
        chunks = chunk_paper(make_paper(REPEATED_TEXT), 500, 50)
        assert chunks[0].text[-50:] == chunks[1].text[:50]
        assert chunks[1].text[-50:] == chunks[2].text[:50]

    def test_short_text_produces_single_chunk(self) -> None:
        chunks = chunk_paper(make_paper(REPEATED_TEXT[:499]), 500, 50)
        assert len(chunks) == 1
        assert chunks[0].char_end == 499

    def test_chunk_index_is_sequential_and_metadata_is_filled(self) -> None:
        chunks = chunk_paper(make_paper(REPEATED_TEXT), 500, 50)
        assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
        for chunk in chunks:
            assert chunk.source == "论文1.pdf"
            assert chunk.source_path == "论文1.pdf"
            assert chunk.file_sha256 == SHA_PLACEHOLDER
            assert len(chunk.id) == 64
            assert chunk.page_start == 1
            assert chunk.page_end == 1

    def test_invalid_parameters_raise(self) -> None:
        paper = make_paper(REPEATED_TEXT)
        with pytest.raises(RagError):
            chunk_paper(paper, 500, 500)
        with pytest.raises(RagError):
            chunk_paper(paper, 500, -1)
        with pytest.raises(RagError):
            chunk_paper(paper, 0, 0)

    def test_ids_are_stable_for_same_input(self) -> None:
        first = chunk_paper(make_paper(REPEATED_TEXT), 500, 50)
        second = chunk_paper(make_paper(REPEATED_TEXT), 500, 50)
        assert [chunk.id for chunk in first] == [chunk.id for chunk in second]

    def test_ids_change_when_file_hash_changes(self) -> None:
        first = chunk_paper(make_paper(REPEATED_TEXT), 500, 50)
        second = chunk_paper(make_paper(REPEATED_TEXT, file_sha256="b" * 64), 500, 50)
        assert [chunk.id for chunk in first] != [chunk.id for chunk in second]

    def test_cross_page_chunk_reports_both_pages(self) -> None:
        chunks = chunk_paper(make_two_page_paper(), 500, 50)
        assert chunks[0].page_start == 1
        assert chunks[0].page_end == 2
        assert chunks[-1].page_end == 2

    def test_chunk_inside_second_page_reports_single_page(self) -> None:
        chunks = chunk_paper(make_two_page_paper(), 200, 0)
        assert chunks[3].page_start == 2
        assert chunks[3].page_end == 2

    def test_blank_only_text_produces_no_chunk(self) -> None:
        assert chunk_paper(make_paper("   \n\n  "), 500, 50) == []


class TestPagesForRange:
    def test_returns_first_and_last_matching_page(self) -> None:
        spans = (
            PageSpan(page_number=1, start=0, end=300),
            PageSpan(page_number=2, start=302, end=602),
        )
        assert pages_for_range(spans, 250, 350) == (1, 2)
        assert pages_for_range(spans, 0, 300) == (1, 1)
        assert pages_for_range(spans, 302, 602) == (2, 2)

    def test_no_match_raises_instead_of_faking_page_zero(self) -> None:
        spans = (PageSpan(page_number=1, start=0, end=10),)
        with pytest.raises(RagError):
            pages_for_range(spans, 20, 30)


class TestSha256File:
    def test_hash_is_stable_hex_digest(self, tmp_path: Path) -> None:
        target = tmp_path / "论文1.pdf"
        target.write_bytes(b"hello world")
        digest = sha256_file(target)
        assert len(digest) == 64
        assert digest == digest.lower()
        assert sha256_file(target) == digest

    def test_missing_file_raises_with_path(self, tmp_path: Path) -> None:
        with pytest.raises(RagError, match="无法读取文件"):
            sha256_file(tmp_path / "missing.pdf")


class TestDiscoverPdfs:
    def test_finds_pdf_regardless_of_case_and_sorts_stably(self, tmp_path: Path) -> None:
        (tmp_path / "b.pdf").write_bytes(b"b")
        (tmp_path / "a.PDF").write_bytes(b"a")
        (tmp_path / "c.pdf").write_bytes(b"c")
        nested = tmp_path / "sub"
        nested.mkdir()
        (nested / "d.pdf").write_bytes(b"d")

        found = discover_pdfs(tmp_path)
        assert [path.name for path in found] == ["a.PDF", "b.pdf", "c.pdf", "d.pdf"]

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(RagError, match="不存在"):
            discover_pdfs(tmp_path / "absent")

    def test_file_instead_of_directory_raises(self, tmp_path: Path) -> None:
        target = tmp_path / "not-a-dir"
        target.write_bytes(b"x")
        with pytest.raises(RagError, match="不是目录"):
            discover_pdfs(target)

    def test_directory_without_pdf_raises(self, tmp_path: Path) -> None:
        (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")
        with pytest.raises(RagError, match="没有找到 PDF"):
            discover_pdfs(tmp_path)


class TestExtractPdf:
    def test_extracts_text_and_one_based_pages(self, tmp_path: Path) -> None:
        papers_dir = tmp_path / "papers"
        papers_dir.mkdir()
        pdf_path = papers_dir / "sample.pdf"
        write_pdf(pdf_path, ["Hello world page one.", "Second page content here."])

        paper = extract_pdf(pdf_path, papers_dir)

        assert paper.relative_path == "sample.pdf"
        assert "Hello world page one." in paper.text
        assert "Second page content here." in paper.text
        assert [span.page_number for span in paper.page_spans] == [1, 2]
        assert paper.page_spans[0].start == 0
        assert paper.text[: paper.page_spans[0].end].strip() == "Hello world page one."
        assert paper.text[paper.page_spans[1].start :].startswith("Second page content here.")
        assert paper.page_spans[1].start == paper.page_spans[0].end + 2
        assert len(paper.file_sha256) == 64

    def test_pages_are_joined_by_two_newlines(self, tmp_path: Path) -> None:
        papers_dir = tmp_path / "papers"
        papers_dir.mkdir()
        pdf_path = papers_dir / "sample.pdf"
        write_pdf(pdf_path, ["Alpha page.", "Beta page."])

        paper = extract_pdf(pdf_path, papers_dir)
        assert "\n\n" in paper.text

    def test_pdf_without_text_layer_mentions_ocr(self, tmp_path: Path) -> None:
        papers_dir = tmp_path / "papers"
        papers_dir.mkdir()
        pdf_path = papers_dir / "scanned.pdf"
        document = fitz.open()
        try:
            document.new_page()
            document.save(pdf_path)
        finally:
            document.close()

        with pytest.raises(RagError, match="扫描版|OCR"):
            extract_pdf(pdf_path, papers_dir)

    def test_encrypted_pdf_reports_encryption(self, tmp_path: Path) -> None:
        papers_dir = tmp_path / "papers"
        papers_dir.mkdir()
        pdf_path = papers_dir / "locked.pdf"
        document = fitz.open()
        try:
            page = document.new_page()
            page.insert_text((72, 72), "secret")
            document.save(pdf_path, encryption=fitz.PDF_ENCRYPT_AES_128, owner_pw="pw", user_pw="pw")
        finally:
            document.close()

        with pytest.raises(RagError, match="加密"):
            extract_pdf(pdf_path, papers_dir)


class TestChunkRecordContract:
    def test_chunk_record_fields_are_complete(self) -> None:
        chunks = chunk_paper(make_paper(REPEATED_TEXT), 500, 50)
        record = chunks[0]
        assert isinstance(record, ChunkRecord)
        assert record.text == REPEATED_TEXT[0:500]
        assert record.char_start == 0
        assert record.char_end == 500
        assert record.chunk_index == 0


def make_config(tmp_path: Path, **overrides: object) -> Config:
    """构造测试用 Config，默认值与生产默认值一致。"""

    values: dict[str, object] = dict(
        papers_dir=tmp_path / "papers",
        chroma_dir=tmp_path / "chroma_db",
        collection_name="science_papers",
        embedding_model="intfloat/multilingual-e5-small",
        ollama_base_url="http://127.0.0.1:11434",
        ollama_model="deepseek-r1:7b",
        chunk_size=500,
        chunk_overlap=50,
        top_k=3,
        embedding_batch_size=32,
        chroma_write_batch_size=256,
        ollama_connect_timeout=5,
        ollama_read_timeout=300,
    )
    values.update(overrides)
    return Config(**values)  # type: ignore[arg-type]


def make_papers_dir(tmp_path: Path, names: list[str], text: str = LONG_TEXT) -> Path:
    """在临时目录中生成若干内容相同的 PDF。"""

    papers = tmp_path / "papers"
    papers.mkdir(exist_ok=True)
    for name in names:
        write_pdf(papers / name, [text])
    return papers


class FakeEmbeddingModel:
    """记录 encode 调用参数的假 Embedding 模型，不下载真实模型。

    与真实 ``SentenceTransformer`` 一致：传入单个字符串返回一维向量，
    传入字符串列表返回二维向量。
    """

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.calls: list[dict[str, object]] = []

    def encode(self, sentences: object, **kwargs: object) -> object:
        texts = [sentences] if isinstance(sentences, str) else list(sentences)  # type: ignore[arg-type]
        self.calls.append({"texts": texts, "kwargs": kwargs})
        vectors = [[float((index % 5) + 1)] * self.dim for index in range(len(texts))]
        return vectors[0] if isinstance(sentences, str) else vectors


class MismatchEmbeddingModel(FakeEmbeddingModel):
    """返回向量数与文本数不一致的假模型。"""

    def encode(self, texts: list[str], **kwargs: object) -> list[list[float]]:
        super().encode(texts, **kwargs)
        return [[1.0] * self.dim]


class TestEmbedPassages:
    def test_uses_passage_prefix_and_normalization(self) -> None:
        model = FakeEmbeddingModel()
        chunks = chunk_paper(make_paper(REPEATED_TEXT), 500, 50)

        vectors = rag.embed_passages(model, chunks, 32)

        assert len(vectors) == len(chunks)
        assert all(len(vector) == 4 for vector in vectors)
        recorded = model.calls[0]
        assert all(str(text).startswith("passage: ") for text in recorded["texts"])  # type: ignore[index]
        assert recorded["kwargs"]["normalize_embeddings"] is True  # type: ignore[index]
        assert recorded["kwargs"]["batch_size"] == 32  # type: ignore[index]

    def test_empty_chunks_raise(self) -> None:
        with pytest.raises(RagError):
            rag.embed_passages(FakeEmbeddingModel(), [], 32)

    def test_invalid_batch_size_raises(self) -> None:
        chunks = chunk_paper(make_paper(REPEATED_TEXT), 500, 50)
        with pytest.raises(RagError):
            rag.embed_passages(FakeEmbeddingModel(), chunks, 0)

    def test_vector_count_mismatch_raises(self) -> None:
        chunks = chunk_paper(make_paper(REPEATED_TEXT), 500, 50)
        with pytest.raises(RagError, match="不一致"):
            rag.embed_passages(MismatchEmbeddingModel(), chunks, 32)


class TestChromaHelpers:
    def test_collection_names_accepts_strings_and_objects(self) -> None:
        class FakeCollection:
            def __init__(self, name: str) -> None:
                self.name = name

        class FakeClient:
            def list_collections(self) -> list[object]:
                return ["alpha", FakeCollection("beta")]

        assert rag.collection_names(FakeClient()) == {"alpha", "beta"}

    def test_create_chroma_client_creates_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "chroma_db"
        client = rag.create_chroma_client(target)
        assert target.is_dir()
        assert "science_papers" not in rag.collection_names(client)

    def test_chunk_metadata_uses_plain_types(self) -> None:
        chunk = chunk_paper(make_paper(REPEATED_TEXT), 500, 50)[0]
        metadata = rag.chunk_metadata(chunk)
        assert metadata["source"] == "论文1.pdf"
        assert metadata["source_path"] == "论文1.pdf"
        assert metadata["page_start"] == 1
        assert metadata["page_end"] == 1
        assert metadata["chunk_index"] == 0
        assert metadata["char_start"] == 0
        assert metadata["char_end"] == 500
        assert metadata["file_sha256"] == SHA_PLACEHOLDER


class TestBuildIndex:
    def test_collection_metadata_documents_and_vectors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        papers = make_papers_dir(tmp_path, ["论文1.pdf", "论文2.pdf"])
        config = make_config(tmp_path, papers_dir=papers)
        monkeypatch.setattr(rag, "load_embedding_model", lambda name: FakeEmbeddingModel())

        rag.build_index(config)

        client = chromadb.PersistentClient(path=str(config.chroma_dir))
        collection = client.get_collection("science_papers")
        assert collection.count() == 6

        metadata = collection.metadata or {}
        assert metadata["hnsw:space"] == "cosine"
        assert metadata["embedding_model"] == "intfloat/multilingual-e5-small"
        assert metadata["chunk_size"] == 500
        assert metadata["chunk_overlap"] == 50

        fetched = collection.get(include=["documents", "metadatas", "embeddings"])
        assert fetched["documents"]
        assert all(not document.startswith("passage:") for document in fetched["documents"])
        assert fetched["embeddings"] is not None
        assert all(len(vector) == 4 for vector in fetched["embeddings"])
        for item in fetched["metadatas"] or []:
            assert item["source"] in {"论文1.pdf", "论文2.pdf"}
            assert item["source_path"] in {"论文1.pdf", "论文2.pdf"}
            assert item["page_start"] >= 1
            assert item["page_end"] >= item["page_start"]
            assert item["chunk_index"] >= 0
            assert len(item["file_sha256"]) == 64

    def test_full_rebuild_replaces_stale_chunks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        papers = make_papers_dir(tmp_path, ["论文1.pdf"])
        config = make_config(tmp_path, papers_dir=papers)
        monkeypatch.setattr(rag, "load_embedding_model", lambda name: FakeEmbeddingModel())

        rag.build_index(config)
        client = chromadb.PersistentClient(path=str(config.chroma_dir))
        assert client.get_collection("science_papers").count() == 3

        (papers / "论文1.pdf").unlink()
        write_pdf(papers / "论文2.pdf", ["A short paper."])
        rag.build_index(config)

        client = chromadb.PersistentClient(path=str(config.chroma_dir))
        collection = client.get_collection("science_papers")
        assert collection.count() == 1
        assert collection.get(include=["metadatas"])["metadatas"][0]["source"] == "论文2.pdf"

    def test_one_failing_pdf_still_indexes_the_rest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        papers = make_papers_dir(tmp_path, ["论文1.pdf", "论文2.pdf"])
        config = make_config(tmp_path, papers_dir=papers)
        real_extract = rag.extract_pdf

        def fake_extract(path: Path, papers_dir: Path) -> ExtractedPaper:
            if path.name == "论文2.pdf":
                raise RagError("模拟提取失败")
            return real_extract(path, papers_dir)

        monkeypatch.setattr(rag, "extract_pdf", fake_extract)
        monkeypatch.setattr(rag, "load_embedding_model", lambda name: FakeEmbeddingModel())

        rag.build_index(config)

        client = chromadb.PersistentClient(path=str(config.chroma_dir))
        fetched = client.get_collection("science_papers").get(include=["metadatas"])
        assert {item["source"] for item in fetched["metadatas"] or []} == {"论文1.pdf"}

    def test_all_pdfs_failing_keeps_previous_collection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        papers = make_papers_dir(tmp_path, ["论文1.pdf"])
        config = make_config(tmp_path, papers_dir=papers)
        monkeypatch.setattr(rag, "load_embedding_model", lambda name: FakeEmbeddingModel())
        rag.build_index(config)

        client = chromadb.PersistentClient(path=str(config.chroma_dir))
        assert client.get_collection("science_papers").count() == 3

        def always_fail(path: Path, papers_dir: Path) -> ExtractedPaper:
            raise RagError("模拟全部失败")

        monkeypatch.setattr(rag, "extract_pdf", always_fail)
        with pytest.raises(RagError, match="所有 PDF 都提取失败"):
            rag.build_index(config)

        client = chromadb.PersistentClient(path=str(config.chroma_dir))
        assert client.get_collection("science_papers").count() == 3

    def test_missing_papers_directory_raises(self, tmp_path: Path) -> None:
        config = make_config(tmp_path)
        with pytest.raises(RagError, match="不存在"):
            rag.build_index(config)


class FakeCollection:
    """记录 query 调用参数的假 Chroma collection。"""

    def __init__(
        self,
        count: int,
        rows: list[dict[str, object]] | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self._count = count
        self.rows = rows or []
        self.metadata = metadata
        self.query_calls: list[dict[str, object]] = []

    def count(self) -> int:
        return self._count

    def query(self, **kwargs: object) -> dict[str, list[list[object]]]:
        self.query_calls.append(kwargs)
        return {
            "documents": [[row["text"] for row in self.rows]],
            "metadatas": [[row["metadata"] for row in self.rows]],
            "distances": [[row["distance"] for row in self.rows]],
        }


class FakeClient:
    """只实现 get_index_collection 需要接口的假 Chroma 客户端。"""

    def __init__(self, collection: FakeCollection | None, names: set[str]) -> None:
        self._collection = collection
        self._names = names

    def list_collections(self) -> list[str]:
        return sorted(self._names)

    def get_collection(self, name: str) -> FakeCollection:
        assert self._collection is not None
        return self._collection


def make_rows(count: int) -> list[dict[str, object]]:
    """生成 count 条结构完整的假检索结果。"""

    return [
        {
            "text": f"chunk number {index}",
            "metadata": {
                "source": f"论文{index}.pdf",
                "source_path": f"论文{index}.pdf",
                "page_start": index + 1,
                "page_end": index + 2,
                "chunk_index": index,
            },
            "distance": 0.1 * (index + 1),
        }
        for index in range(count)
    ]


class FakeResponse:
    """可控制的假 requests.Response。"""

    def __init__(
        self,
        *,
        status_code: int = 200,
        json_data: object = None,
        text: str = "",
        invalid_json: bool = False,
    ) -> None:
        self.status_code = status_code
        self._json_data = json_data
        self.text = text
        self._invalid_json = invalid_json

    def json(self) -> object:
        if self._invalid_json:
            raise ValueError("Expecting value: line 1 column 1 (char 0)")
        return self._json_data


class TestEmbedQuery:
    def test_uses_query_prefix_and_strips_whitespace(self) -> None:
        model = FakeEmbeddingModel()
        vector = rag.embed_query(model, "  这些论文研究了什么问题？  ")

        assert len(vector) == 4
        assert model.calls[0]["texts"] == ["query: 这些论文研究了什么问题？"]
        assert model.calls[0]["kwargs"]["normalize_embeddings"] is True
        assert model.calls[0]["kwargs"]["show_progress_bar"] is False

    def test_empty_question_raises(self) -> None:
        with pytest.raises(RagError, match="问题不能为空"):
            rag.embed_query(FakeEmbeddingModel(), "   \n  ")


class TestGetIndexCollection:
    def test_missing_collection_asks_for_index(self, tmp_path: Path) -> None:
        client = FakeClient(None, set())
        config = make_config(tmp_path)
        with pytest.raises(RagError, match="python rag.py index"):
            rag.get_index_collection(client, config)

    def test_empty_collection_asks_for_index(self, tmp_path: Path) -> None:
        client = FakeClient(FakeCollection(0, []), {"science_papers"})
        config = make_config(tmp_path)
        with pytest.raises(RagError, match="python rag.py index"):
            rag.get_index_collection(client, config)

    def test_model_mismatch_asks_for_rebuild(self, tmp_path: Path) -> None:
        collection = FakeCollection(3, make_rows(3), {"embedding_model": "other-model"})
        client = FakeClient(collection, {"science_papers"})
        config = make_config(tmp_path)
        with pytest.raises(RagError, match="重建索引"):
            rag.get_index_collection(client, config)

    def test_chunk_parameter_mismatch_asks_for_rebuild(self, tmp_path: Path) -> None:
        metadata = {
            "embedding_model": "intfloat/multilingual-e5-small",
            "chunk_size": 300,
            "chunk_overlap": 30,
        }
        collection = FakeCollection(3, make_rows(3), metadata)
        client = FakeClient(collection, {"science_papers"})
        config = make_config(tmp_path)
        with pytest.raises(RagError, match="python rag.py index"):
            rag.get_index_collection(client, config)

    def test_matching_configuration_returns_collection(self, tmp_path: Path) -> None:
        metadata = {
            "embedding_model": "intfloat/multilingual-e5-small",
            "chunk_size": 500,
            "chunk_overlap": 50,
        }
        collection = FakeCollection(3, make_rows(3), metadata)
        client = FakeClient(collection, {"science_papers"})
        config = make_config(tmp_path)
        assert rag.get_index_collection(client, config) is collection


class TestRetrieve:
    def test_requests_exactly_top_k_when_more_available(self) -> None:
        collection = FakeCollection(10, make_rows(3))
        hits = rag.retrieve(collection, [0.1, 0.2], 3)

        assert collection.query_calls[0]["n_results"] == 3
        assert collection.query_calls[0]["include"] == ["documents", "metadatas", "distances"]
        assert collection.query_calls[0]["query_embeddings"] == [[0.1, 0.2]]
        assert [hit.rank for hit in hits] == [1, 2, 3]
        assert [hit.distance for hit in hits] == pytest.approx([0.1, 0.2, 0.3])
        assert [hit.text for hit in hits] == ["chunk number 0", "chunk number 1", "chunk number 2"]

    def test_returns_all_results_when_fewer_than_top_k(self) -> None:
        collection = FakeCollection(2, make_rows(2))
        hits = rag.retrieve(collection, [0.1, 0.2], 3)
        assert collection.query_calls[0]["n_results"] == 2
        assert len(hits) == 2

    def test_empty_collection_raises(self) -> None:
        collection = FakeCollection(0, [])
        with pytest.raises(RagError, match="python rag.py index"):
            rag.retrieve(collection, [0.1], 3)

    def test_invalid_top_k_raises(self) -> None:
        collection = FakeCollection(3, make_rows(3))
        with pytest.raises(RagError):
            rag.retrieve(collection, [0.1], 0)

    def test_missing_metadata_field_raises(self) -> None:
        rows = make_rows(1)
        del rows[0]["metadata"]["source"]  # type: ignore[index]
        collection = FakeCollection(1, rows)
        with pytest.raises(RagError, match="元数据"):
            rag.retrieve(collection, [0.1], 3)

    def test_inconsistent_result_lengths_raise(self) -> None:
        class BrokenCollection(FakeCollection):
            def query(self, **kwargs: object) -> dict[str, list[list[object]]]:
                return {
                    "documents": [["a", "b"]],
                    "metadatas": [[{}, {}]],
                    "distances": [[0.1]],
                }

        with pytest.raises(RagError, match="不一致"):
            rag.retrieve(BrokenCollection(2), [0.1], 2)


class TestFormatPageRange:
    def test_single_and_cross_page(self) -> None:
        assert rag.format_page_range(3, 3) == "第 3 页"
        assert rag.format_page_range(3, 4) == "第 3-4 页"


class TestBuildOllamaMessages:
    def test_prompt_contains_question_and_all_chunks(self) -> None:
        hits = rag.retrieve(FakeCollection(10, make_rows(3)), [0.1], 3)
        messages = rag.build_ollama_messages("这些论文研究了什么问题？", hits)

        assert [message["role"] for message in messages] == ["system", "user"]
        user = messages[1]["content"]
        assert "这些论文研究了什么问题？" in user
        for index in range(3):
            assert f"chunk number {index}" in user
            assert f"[{index + 1}]" in user
            assert f"论文{index}.pdf" in user
            assert f"第 {index + 1}-{index + 2} 页" in user

    def test_prompt_hides_distance(self) -> None:
        hits = rag.retrieve(FakeCollection(10, make_rows(3)), [0.1], 3)
        user = rag.build_ollama_messages("问题", hits)[1]["content"]
        assert "0.1" not in user
        assert "distance" not in user

    def test_system_prompt_states_grounding_rules(self) -> None:
        hits = rag.retrieve(FakeCollection(10, make_rows(3)), [0.1], 3)
        system = rag.build_ollama_messages("问题", hits)[0]["content"]
        assert "仅依据片段" in system
        assert "证据不足" in system

    def test_empty_hits_raises(self) -> None:
        with pytest.raises(RagError):
            rag.build_ollama_messages("问题", [])


class TestCallOllama:
    def test_posts_expected_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def fake_post(url: str, **kwargs: object) -> FakeResponse:
            captured["url"] = url
            captured["kwargs"] = kwargs
            return FakeResponse(json_data={"message": {"content": "最终回答 [1]。"}})

        monkeypatch.setattr(rag.requests, "post", fake_post)
        config = make_config(Path("."))

        answer = rag.call_ollama([{"role": "user", "content": "hi"}], config)

        assert answer == "最终回答 [1]。"
        assert captured["url"] == "http://127.0.0.1:11434/api/chat"
        payload = captured["kwargs"]["json"]  # type: ignore[index]
        assert payload["model"] == "deepseek-r1:7b"
        assert payload["stream"] is False
        assert payload["options"]["temperature"] == 0.2
        assert captured["kwargs"]["timeout"] == (5, 300)  # type: ignore[index]

    def test_connection_error_mentions_ollama_serve(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_post(url: str, **kwargs: object):
            raise requests.ConnectionError("refused")

        monkeypatch.setattr(rag.requests, "post", fake_post)
        with pytest.raises(RagError, match="ollama serve"):
            rag.call_ollama([{"role": "user", "content": "hi"}], make_config(Path(".")))

    def test_timeout_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_post(url: str, **kwargs: object):
            raise requests.Timeout("timed out")

        monkeypatch.setattr(rag.requests, "post", fake_post)
        with pytest.raises(RagError, match="超时"):
            rag.call_ollama([{"role": "user", "content": "hi"}], make_config(Path(".")))

    def test_missing_model_suggests_pull_command(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            rag.requests,
            "post",
            lambda url, **kwargs: FakeResponse(
                status_code=404,
                json_data={"error": "model 'definitely-not-installed' not found"},
            ),
        )
        with pytest.raises(RagError, match="ollama pull deepseek-r1:7b"):
            rag.call_ollama([{"role": "user", "content": "hi"}], make_config(Path(".")))

    def test_pull_hint_follows_overridden_model_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            rag.requests,
            "post",
            lambda url, **kwargs: FakeResponse(
                status_code=404,
                json_data={"error": "model 'other-model' not found"},
            ),
        )
        config = make_config(Path("."), ollama_model="other-model")
        with pytest.raises(RagError, match="ollama pull other-model"):
            rag.call_ollama([{"role": "user", "content": "hi"}], config)

    def test_invalid_json_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            rag.requests,
            "post",
            lambda url, **kwargs: FakeResponse(status_code=200, invalid_json=True),
        )
        with pytest.raises(RagError, match="JSON"):
            rag.call_ollama([{"role": "user", "content": "hi"}], make_config(Path(".")))

    def test_empty_content_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            rag.requests,
            "post",
            lambda url, **kwargs: FakeResponse(json_data={"message": {"content": "   "}}),
        )
        with pytest.raises(RagError, match="空回答"):
            rag.call_ollama([{"role": "user", "content": "hi"}], make_config(Path(".")))

    def test_missing_message_field_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            rag.requests,
            "post",
            lambda url, **kwargs: FakeResponse(json_data={"done": True}),
        )
        with pytest.raises(RagError, match="message"):
            rag.call_ollama([{"role": "user", "content": "hi"}], make_config(Path(".")))

    def test_thinking_field_is_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            rag.requests,
            "post",
            lambda url, **kwargs: FakeResponse(
                json_data={"thinking": "内部推理", "message": {"content": "最终答案"}}
            ),
        )
        assert rag.call_ollama([], make_config(Path("."))) == "最终答案"


class TestCleanModelAnswer:
    def test_removes_think_tags(self) -> None:
        raw = "<think>先想很久</think>最终答案在这里。"
        assert rag.clean_model_answer(raw) == "最终答案在这里。"

    def test_removes_multiline_think_tags(self) -> None:
        raw = "<think>第一行\n第二行</think>\n最终答案。\n"
        assert rag.clean_model_answer(raw) == "最终答案。"

    def test_thinking_only_content_raises(self) -> None:
        with pytest.raises(RagError, match="思考"):
            rag.clean_model_answer("<think>只有思考，没有答案</think>")

    def test_plain_answer_is_kept(self) -> None:
        assert rag.clean_model_answer("  直接回答。  ") == "直接回答。"


class TestCitedSources:
    def test_deduplicates_by_source_path_and_keeps_rank_order(self) -> None:
        rows = make_rows(3)
        rows[1]["metadata"]["source"] = "论文1.pdf"
        rows[1]["metadata"]["source_path"] = "论文1.pdf"
        hits = rag.retrieve(FakeCollection(3, rows), [0.1], 3)

        assert rag.cited_sources(hits) == ["论文0.pdf", "论文1.pdf", "论文2.pdf"]

    def test_repeated_paper_appears_once(self) -> None:
        rows = make_rows(3)
        for row in rows:
            row["metadata"]["source"] = "论文1.pdf"  # type: ignore[index]
            row["metadata"]["source_path"] = "论文1.pdf"  # type: ignore[index]
        hits = rag.retrieve(FakeCollection(3, rows), [0.1], 3)

        assert rag.cited_sources(hits) == ["论文1.pdf"]


class TestAnswerQuestion:
    def _prepare(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Config:
        papers = make_papers_dir(tmp_path, ["论文1.pdf", "论文4.pdf"])
        config = make_config(tmp_path, papers_dir=papers)
        monkeypatch.setattr(rag, "load_embedding_model", lambda name: FakeEmbeddingModel())
        rag.build_index(config)
        return config

    def test_prints_answer_and_deduplicated_citations(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        config = self._prepare(tmp_path, monkeypatch)

        rows = make_rows(3)
        for row in rows[:2]:
            row["metadata"]["source"] = "论文1.pdf"  # type: ignore[index]
            row["metadata"]["source_path"] = "论文1.pdf"  # type: ignore[index]
        rows[2]["metadata"]["source"] = "论文4.pdf"  # type: ignore[index]
        rows[2]["metadata"]["source_path"] = "论文4.pdf"  # type: ignore[index]
        hits = rag.retrieve(FakeCollection(3, rows), [0.1], 3)
        monkeypatch.setattr(rag, "retrieve", lambda collection, embedding, top_k: hits)

        def fake_post(url: str, **kwargs: object) -> FakeResponse:
            return FakeResponse(
                json_data={
                    "thinking": "内部推理过程",
                    "message": {"content": "根据片段可以确定结论 [1][2]。"},
                }
            )

        monkeypatch.setattr(rag.requests, "post", fake_post)

        rag.answer_question("这些论文主要研究了什么问题？", config)
        output = capsys.readouterr().out

        assert "回答：" in output
        assert "根据片段可以确定结论 [1][2]。" in output
        assert "内部推理过程" not in output
        assert "引用论文：" in output
        assert output.count("- 论文1.pdf") == 1
        assert "- 论文4.pdf" in output

    def test_blank_question_raises_without_touching_services(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = make_config(tmp_path)

        def forbidden(*args: object, **kwargs: object):
            raise AssertionError("不应在问题为空时调用外部服务")

        monkeypatch.setattr(rag, "create_chroma_client", forbidden)
        monkeypatch.setattr(rag, "load_embedding_model", forbidden)

        with pytest.raises(RagError, match="问题不能为空"):
            rag.answer_question("   ", config)


class TestCliBehaviour:
    def test_ask_with_empty_string_returns_non_zero(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        assert rag.main(["ask", ""]) == 1
        assert "错误：问题不能为空" in capsys.readouterr().err

    def test_ask_without_question_prompts_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("builtins.input", lambda prompt: "一次交互输入的问题")

        captured: dict[str, str] = {}

        def fake_answer(question: str, config: object) -> None:
            captured["question"] = question

        monkeypatch.setattr(rag, "answer_question", fake_answer)
        assert rag.main(["ask"]) == 0
        assert captured["question"] == "一次交互输入的问题"

    def test_keyboard_interrupt_returns_130(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        def interrupt() -> rag.Config:
            raise KeyboardInterrupt

        monkeypatch.setattr(rag.Config, "from_env", interrupt)
        assert rag.main(["ask", "问题"]) == 130
        assert "已取消。" in capsys.readouterr().err

    def test_config_error_is_reported_in_chinese(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        monkeypatch.setenv("CHUNK_OVERLAP", "500")
        monkeypatch.setenv("CHUNK_SIZE", "500")
        assert rag.main(["index"]) == 1
        assert "错误：" in capsys.readouterr().err
