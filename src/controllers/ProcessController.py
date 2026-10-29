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
from dataclasses import dataclass

@dataclass
class Document:
    page_content: str
    metadata : dict

class ProcessController(BaseController):
    def __init__(self, project_id:str):
        super().__init__()

        self.project_id = project_id
        self.project_path = ProjectController().get_project_path(project_id=project_id)

    def get_file_extension(self, file_id:str):
         return os.path.splitext(file_id)[-1]   

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

        if file_extension == ProcessingSgnal.CSV.value:
            chunks = []
            row_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, 
                chunk_overlap=overlap_size
            )
            for doc in file_content:
                meta = dict(doc.metadata) if doc.metadata else {}
                if "row" in meta:
                    meta["row"] = meta["row"] + 1
                meta["source_format"] = "csv"

                if len(doc.page_content) <= chunk_size:
                    chunks.append(Document(page_content=doc.page_content, metadata=meta))
                else:
                    sub_docs = row_splitter.split_documents([Document(page_content=doc.page_content, metadata=meta)])
                    chunks.extend(sub_docs)

            return chunks

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap_size,
            length_function=len
        )

        chunks = text_splitter.split_documents(file_content)

        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = idx + 1
            if file_extension == ProcessingSgnal.PDF.value:
                chunk.metadata["source_format"] = "pdf"

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
