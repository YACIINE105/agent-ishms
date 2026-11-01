from .BaseController import BaseController
from .ProjectController import ProjectController
import os
from langchain_community.document_loaders import TextLoader
from ingestion import QalamLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from models.enums import ProcessingSgnal
from typing import List
class ProcessController(BaseController):
    def __init__(self, project_id:str):
        super().__init__()

        self.project_id = project_id
        self.project_path = ProjectController().get_project_path(project_id=project_id)

    def get_file_extension(self, file_id:str):
        return os.path.splitext(file_id)[-1].lower()

    def get_file_loader(self, file_id:str):

        file_extension = self.get_file_extension(file_id=file_id)
        file_path = os.path.join(self.project_path, file_id)

        if not os.path.exists(file_path):
            return None

        if file_extension == ProcessingSgnal.TXT.value:
            return TextLoader(file_path=file_path, encoding="utf-8")

        if file_extension == ProcessingSgnal.PDF.value:
            return QalamLoader(file_path=file_path)

        if file_extension == ProcessingSgnal.CSV.value:
            return CSVLoader(
                        file_path=file_path,
                        encoding="utf-8-sig"
                                                )

        return None

    def get_file_content(self, file_id:str):
        loader = self.get_file_loader(file_id=file_id)
        if loader:
            return loader.load()
        else:
            return None

    def get_file_content_2(self, file_id:str):
        loader = self.get_file_loader(file_id=file_id)
        if loader:
            return loader.document
        else:
            return None

    def process_file_content(self, file_content: list, file_id: str, chunk_size: int = 400, overlap_size: int = 50):
        if not file_content:
            return []

        file_extension = self.get_file_extension(file_id=file_id).lower()
        chunks = []

        for doc_idx, doc in enumerate(file_content):
            raw_metadata = doc.metadata if hasattr(doc, "metadata") and doc.metadata else {}
            content = doc.page_content if hasattr(doc, "page_content") else str(doc)

            # Base provenance contract: source_document, file_format, page_number (1-based)
            raw_page = raw_metadata.get("page_number") or raw_metadata.get("page")
            page_num = int(raw_page) if (raw_page is not None and str(raw_page).isdigit()) else None

            source_format = raw_metadata.get("file_format") or raw_metadata.get("source_format")
            if not source_format:
                source_format = file_extension.lstrip(".") if file_extension else "unknown"

            base_metadata = {
                "source_document": os.path.basename(file_id),
                "source_name": os.path.basename(file_id),
                "file_format": source_format,
                "source_format": source_format,
                "page_number": page_num,
                "page": page_num,
            }

            if "verdict" in raw_metadata:
                base_metadata["verdict"] = raw_metadata["verdict"]
            if "confidence" in raw_metadata:
                base_metadata["confidence"] = raw_metadata["confidence"]
            if "reasons" in raw_metadata:
                base_metadata["reasons"] = raw_metadata["reasons"]
            if "needs_ocr" in raw_metadata:
                base_metadata["needs_ocr"] = raw_metadata["needs_ocr"]

            # 4- CSV Row Handling: Preserves each CSV row as an atomic record with 1-based row_index
            if "row" in raw_metadata or file_extension == ProcessingSgnal.CSV.value:
                csv_row = raw_metadata.get("row", doc_idx)
                row_index = int(csv_row) + 1  # 1-based row index
                row_metadata = {
                    **base_metadata,
                    "row_index": row_index,
                    "chunk_index": len(chunks) + 1
                }
                chunks.append(Document(page_content=content.strip(), metadata=row_metadata))
                continue

            # 2- Table Preservation: Atomic Table Chunks from QalamLoader
            doc_tables = raw_metadata.get("tables", [])
            if doc_tables:
                table_rendered_rows = {}
                for tbl in doc_tables:
                    tbl_id = tbl.get("table_id")
                    tbl_rows = tbl.get("rows", [])
                    for r in tbl_rows:
                        r_text = " | ".join(r)
                        table_rendered_rows[r_text] = tbl_id

                lines = [l.strip() for l in content.split("\n") if l.strip()]
                prose_buffer = []
                current_table_id = None
                current_table_rows = []

                def flush_prose():
                    nonlocal prose_buffer
                    if prose_buffer:
                        prose_text = "\n".join(prose_buffer)
                        prose_chunks = self.split_prose_text(prose_text, chunk_size=chunk_size, base_metadata=base_metadata)
                        for pc in prose_chunks:
                            pc.metadata["chunk_index"] = len(chunks) + 1
                            chunks.append(pc)
                        prose_buffer = []

                def flush_table():
                    nonlocal current_table_rows, current_table_id
                    if current_table_rows and current_table_id:
                        tbl_text = "\n".join(current_table_rows)
                        tbl_metadata = {
                            **base_metadata,
                            "table_id": current_table_id,
                            "chunk_index": len(chunks) + 1
                        }
                        chunks.append(Document(page_content=tbl_text, metadata=tbl_metadata))
                        current_table_rows = []
                        current_table_id = None

                for line in lines:
                    matched_tbl_id = table_rendered_rows.get(line)
                    if matched_tbl_id:
                        flush_prose()
                        if current_table_id and current_table_id != matched_tbl_id:
                            flush_table()
                        current_table_id = matched_tbl_id
                        current_table_rows.append(line)
                    else:
                        flush_table()
                        prose_buffer.append(line)

                flush_prose()
                flush_table()
            else:
                # Standard prose document (TXT or PDF without detected tables)
                prose_chunks = self.split_prose_text(content, chunk_size=chunk_size, base_metadata=base_metadata)
                for pc in prose_chunks:
                    pc.metadata["chunk_index"] = len(chunks) + 1
                    chunks.append(pc)

        return chunks

    def split_prose_text(self, text: str, chunk_size: int, base_metadata: dict, splitter_tag: str = "\n") -> List[Document]:
        lines = [doc.strip() for doc in text.split(splitter_tag) if len(doc.strip()) > 0]
        chunks = []
        current_chunk = ""

        for line in lines:
            while len(line) > chunk_size:
                if current_chunk:
                    chunks.append(Document(page_content=current_chunk.strip(), metadata=dict(base_metadata)))
                    current_chunk = ""
                chunks.append(Document(page_content=line[:chunk_size].strip(), metadata=dict(base_metadata)))
                line = line[chunk_size:].strip()
            if current_chunk and len(current_chunk) + len(line) + 1 > chunk_size:
                chunks.append(Document(page_content=current_chunk.strip(), metadata=dict(base_metadata)))
                current_chunk = line + splitter_tag
            else:
                current_chunk += line + splitter_tag

        if len(current_chunk.strip()) > 0:
            chunks.append(Document(page_content=current_chunk.strip(), metadata=dict(base_metadata)))

        return chunks

    def process_simple_splitter(self, texts:List[str], metadatas:List[dict], chunk_size:int=100, splitter_tag:str="\n"):
        full_text = "".join(texts)

        lines = [doc.strip() for doc in full_text.split(splitter_tag) if len(doc.strip()) > 1] 

        chunks = []
        current_chunk = ""

        for line in lines:
            current_chunk += line + splitter_tag

            if len(current_chunk) >= chunk_size:

                chunks.append(Document(page_content=current_chunk.strip(), 
                                       metadata={}))
                current_chunk = ""

        if len(current_chunk) > 0 :

                        chunks.append(Document(page_content=current_chunk.strip(), 
                                               metadata={}))

        return chunks
