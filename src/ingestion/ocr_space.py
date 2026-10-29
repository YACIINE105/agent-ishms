import io
import logging
from typing import Dict, Any, Tuple
import pymupdf
import requests
from helpers.config import get_settings

logger = logging.getLogger(__name__)

OCR_SPACE_URL = "https://api.ocr.space/parse/image"
MAX_FILE_BYTES = 1024 * 1024

def extract_single_page_bytes(pdf_path: str, page_number_1based: int) -> bytes:
    """Extract a single page from a PDF as in-memory PDF bytes using PyMuPDF."""
    doc = pymupdf.open(pdf_path)
    try:
        page_idx = page_number_1based - 1
        if page_idx < 0 or page_idx >= len(doc):
            raise IndexError(f"Page number {page_number_1based} out of range (1..{len(doc)})")

        single_doc = pymupdf.open()
        single_doc.insert_pdf(doc, from_page=page_idx, to_page=page_idx)
        pdf_bytes = single_doc.tobytes(deflate=True)
        single_doc.close()
        return pdf_bytes
    finally:
        doc.close()

def ocr_space_extract_page(
    pdf_path: str,
    page_number_1based: int,
    api_key: str | None = None,
    engine: str = "2",
    language: str = "auto",
    is_table: bool = True,
    timeout: float = 30.0,
) -> Tuple[str, Dict[str, Any]]:
    """
    Submit a single PDF page to the OCR.space API.

    Returns (extracted_text, metadata_dict).
    """
    if api_key is None:
        settings = get_settings()
        api_key = settings.OCR_SPACE_API_KEY

    if not api_key or not api_key.strip():
        logger.warning(
            f"No OCR_SPACE_API_KEY provided or found in settings. Skipping OCR for page {page_number_1based}."
        )
        return "", {
            "ocr_applied": False,
            "ocr_engine": "ocr_space",
            "ocr_status": "skipped_no_api_key",
            "ocr_error": "OCR_SPACE_API_KEY is not set",
        }

    try:
        page_bytes = extract_single_page_bytes(pdf_path, page_number_1based)
    except Exception as exc:
        logger.error(f"Failed to isolate PDF page {page_number_1based} from {pdf_path}: {exc}")
        return "", {
            "ocr_applied": False,
            "ocr_engine": "ocr_space",
            "ocr_status": "failed",
            "ocr_error": f"Failed to slice page: {str(exc)}",
        }

    if len(page_bytes) > MAX_FILE_BYTES:
        logger.warning(
            f"Page {page_number_1based} size ({len(page_bytes)} bytes) exceeds OCR.space 1 MB free tier limit."
        )
        return "", {
            "ocr_applied": False,
            "ocr_engine": "ocr_space",
            "ocr_status": "failed",
            "ocr_error": f"Payload size ({len(page_bytes)} bytes) exceeds 1 MB limit",
        }

    headers = {
        "apikey": api_key.strip("\"'"),
    }
    payload = {
        "OCREngine": engine,
        "language": language,
        "isTable": "true" if is_table else "false",
        "detectOrientation": "true",
        "scale": "true",
    }
    files = {
        "file": (f"page_{page_number_1based}.pdf", page_bytes, "application/pdf"),
    }

    try:
        response = requests.post(
            OCR_SPACE_URL,
            headers=headers,
            data=payload,
            files=files,
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        logger.error(f"OCR.space request failed for page {page_number_1based}: {exc}")
        return "", {
            "ocr_applied": True,
            "ocr_engine": f"ocr_space_engine_{engine}",
            "ocr_status": "failed",
            "ocr_error": str(exc),
        }

    is_errored = data.get("IsErroredOnProcessing", False)
    exit_code = data.get("OCRExitCode")
    error_msg = data.get("ErrorMessage")
    parsed_results = data.get("ParsedResults") or []

    if is_errored or exit_code not in (1, 2):
        err_detail = error_msg or data.get("ErrorDetails") or f"OCRExitCode: {exit_code}"
        logger.warning(f"OCR.space returned error on page {page_number_1based}: {err_detail}")
        return "", {
            "ocr_applied": True,
            "ocr_engine": f"ocr_space_engine_{engine}",
            "ocr_status": "failed",
            "ocr_error": str(err_detail),
            "ocr_exit_code": exit_code,
        }

    extracted_parts = []
    for item in parsed_results:
        parsed_text = (item.get("ParsedText") or "").strip()
        if parsed_text:
            extracted_parts.append(parsed_text)

    combined_text = "\n\n".join(extracted_parts).strip()
    status = "success" if combined_text else "empty"

    return combined_text, {
        "ocr_applied": True,
        "ocr_engine": f"ocr_space_engine_{engine}",
        "ocr_status": status,
        "ocr_exit_code": exit_code,
        "ocr_error": None if status == "success" else "No text extracted by OCR engine",
    }
