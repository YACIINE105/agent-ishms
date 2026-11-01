import os
import qalam
from langchain_core.documents import Document
from ingestion.ocr_space import ocr_space_extract_page

class QalamLoader:
    def __init__(self, file_path: str, enable_ocr: bool = True, ocr_api_key: str | None = None):
        self.file_path = os.fspath(file_path)
        self.enable_ocr = enable_ocr
        self.ocr_api_key = ocr_api_key

    def load(self):
        pdf_document = qalam.Document(self.file_path)

        documents = []

        for page in pdf_document.pages:
            page_parts = []
            tables = []

            for block in page.blocks:
                if block.kind == "table":
                    table_rows = block.to_rows()
                    table_number = len(tables) + 1
                    table_text = "\n".join(" | ".join(row) for row in table_rows)
                    page_parts.append(table_text)
                    tables.append({
                        "table_id": f"page-{page.number}-table-{table_number}",
                        "rows": table_rows,
                        "row_count": block.row_count,
                        "column_count": block.column_count,
                        "confidence": block.confidence,
                    })
                elif block.text:
                    page_parts.append(block.text)

            page_text = "\n".join(page_parts)

            page_metadata = {
                "source": self.file_path,
                "source_document": os.path.basename(self.file_path),
                "source_name": os.path.basename(self.file_path),
                "file_format": "pdf",
                "source_format": "pdf",
                "page": page.number,
                "page_number": page.number,
                "total_pages": len(pdf_document),
                "parser": "qalam",
                "verdict": page.verdict,
                "confidence": page.confidence,
                "needs_ocr": page.needs_ocr,
                "reasons": list(page.reasons),
                "tables": tables,
                "ocr_applied": False,
            }
            if page.needs_ocr and self.enable_ocr:
                ocr_text, ocr_meta = ocr_space_extract_page(
                    pdf_path=self.file_path,
                    page_number_1based=page.number,
                    api_key=self.ocr_api_key,
                )
                page_metadata.update(ocr_meta)

                if ocr_text:
                    page_text = ocr_text

            documents.append(Document(page_content=page_text, metadata=page_metadata))

        return documents
