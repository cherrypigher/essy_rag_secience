"""rag.py 的单元测试。

所有测试都使用临时目录和内存中的假对象：不访问网络、不下载真实 Embedding 模型、
不读取仓库 papers/ 下的真实论文。
"""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from rag import (
    ChunkRecord,
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
    """用 PyMuPDF 生成一个只含简单英文的 PDF，避免测试环境字体问题。"""

    document = fitz.open()
    try:
        for page_text in pages:
            page = document.new_page()
            page.insert_text((72, 72), page_text)
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
