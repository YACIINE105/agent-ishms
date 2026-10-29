from types import SimpleNamespace
from unittest.mock import patch

import pymupdf
import pytest
import qalam

from ingestion import QalamLoader
from ingestion.ocr_space import ocr_space_extract_page


def test_load_digital_pdf_with_page_metadata(tmp_path):
    file_path = tmp_path / "services.pdf"

    with pymupdf.open() as pdf_document:
        page = pdf_document.new_page()
        page.insert_text((72, 72), "Student card fee: 500 DZD")
        page = pdf_document.new_page()
        page.insert_htmlbox(
            pymupdf.Rect(60, 60, 500, 150),
            '<div dir="rtl">رسوم 200 دينار</div>',
        )
        pdf_document.save(file_path)

    documents = QalamLoader(file_path=file_path).load()

    assert len(documents) == 2
    assert "Student card fee: 500 DZD" in documents[0].page_content
    assert "رسوم" in documents[1].page_content
    assert "200" in documents[1].page_content
    assert "دينار" in documents[1].page_content
    assert "\ufffd" not in documents[1].page_content
    assert [document.metadata["page"] for document in documents] == [1, 2]
    assert all(document.metadata["source"] == str(file_path) for document in documents)
    assert all(document.metadata["total_pages"] == 2 for document in documents)
    assert all(document.metadata["parser"] == "qalam" for document in documents)
    assert all(document.metadata["ocr_applied"] is False for document in documents)


def test_preserve_table_grid_and_degraded_page_metadata(monkeypatch):
    table_rows = [["Service", "Fee (DZD)"], ["Card", "500"], ["Certificate", "200"]]
    table = SimpleNamespace(
        kind="table",
        row_count=3,
        column_count=2,
        confidence=0.9,
        to_rows=lambda: table_rows,
    )
    page = SimpleNamespace(
        number=3,
        verdict="degraded",
        confidence=0.8,
        needs_ocr=False,
        reasons=["Unresolved glyph"],
        blocks=[
            SimpleNamespace(kind="text", text="Service fees"),
            table,
            SimpleNamespace(kind="image", text=""),
        ],
    )

    class ExtractedDocument:
        pages_needing_ocr = []
        pages = [page]

        def __len__(self):
            return 3

    monkeypatch.setattr(qalam, "Document", lambda file_path: ExtractedDocument())

    document = QalamLoader(file_path="services.pdf").load()[0]

    assert document.page_content == "Service fees\nService | Fee (DZD)\nCard | 500\nCertificate | 200"
    assert document.metadata["page"] == 3
    assert document.metadata["verdict"] == "degraded"
    assert document.metadata["reasons"] == ["Unresolved glyph"]
    assert document.metadata["tables"][0]["table_id"] == "page-3-table-1"
    assert document.metadata["tables"][0]["rows"] == table_rows


def test_ocr_fallback_triggered_on_needs_ocr_page(tmp_path, monkeypatch):
    file_path = tmp_path / "scanned-page.pdf"

    with pymupdf.open() as pdf_document:
        page = pdf_document.new_page()
        image = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 20), False)
        image.clear_with(255)
        page.insert_image(pymupdf.Rect(60, 60, 160, 160), pixmap=image)
        pdf_document.save(file_path)

    # Mock OCR.space returning extracted text
    mock_ocr_text = "Official Academic Certificate\nGraduation verified."
    mock_ocr_meta = {
        "ocr_applied": True,
        "ocr_engine": "ocr_space_engine_2",
        "ocr_status": "success",
        "ocr_exit_code": 1,
        "ocr_error": None,
    }

    with patch("ingestion.QalamLoader.ocr_space_extract_page", return_value=(mock_ocr_text, mock_ocr_meta)) as mock_ocr:
        documents = QalamLoader(file_path=file_path, ocr_api_key="test_key").load()

        assert len(documents) == 1
        doc = documents[0]
        assert doc.metadata["page"] == 1
        assert doc.metadata["needs_ocr"] is True
        assert doc.metadata["ocr_applied"] is True
        assert doc.metadata["ocr_status"] == "success"
        assert doc.page_content == mock_ocr_text
        mock_ocr.assert_called_once_with(
            pdf_path=str(file_path),
            page_number_1based=1,
            api_key="test_key",
        )


def test_ocr_fallback_keeps_page_and_flags_on_failure(tmp_path):
    file_path = tmp_path / "scanned-page-fail.pdf"

    with pymupdf.open() as pdf_document:
        page = pdf_document.new_page()
        image = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 20), False)
        image.clear_with(255)
        page.insert_image(pymupdf.Rect(60, 60, 160, 160), pixmap=image)
        pdf_document.save(file_path)

    # When no API key is set, OCR is skipped gracefully and the page is preserved
    documents = QalamLoader(file_path=file_path, ocr_api_key="").load()

    assert len(documents) == 1
    doc = documents[0]
    assert doc.metadata["page"] == 1
    assert doc.metadata["needs_ocr"] is True
    assert doc.metadata["ocr_status"] == "skipped_no_api_key"
    assert doc.metadata["ocr_applied"] is False


def test_missing_pdf_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        QalamLoader(file_path=tmp_path / "missing.pdf").load()


def test_malformed_pdf_raises(tmp_path):
    file_path = tmp_path / "broken.pdf"
    file_path.write_bytes(b"This is not a PDF.")

    with pytest.raises(qalam.QalamError):
        QalamLoader(file_path=file_path).load()
