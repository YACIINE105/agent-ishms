from .BaseController import BaseController
from models.db_schems import Project, DataChunk
from stores.llm.LLMEnums import DocumentTypeEnum
from models.enums import DataBaseEnum
from typing import List
import time, json, asyncio
import logging

class NLPController(BaseController):
    def __init__(self, generation_client, embedding_client, vector_db_client, reranker_client=None, template_parser = None):
        super().__init__()

        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.vector_db_client = vector_db_client
        self.template_parser = template_parser
        self.reranker_client = reranker_client
        self.logger = logging.getLogger('uvicorn.error')

    def create_collection_name(self, project_id:str):
        return f"collection_{self.vector_db_client.default_vector_size}_{project_id}".strip()

    async def reset_vector_db_colection(self,project:Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        await self.vector_db_client.delete_collection(collection_name=collection_name)

    async def get_vector_db_collection_info(self, project:Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        collection_info = await self.vector_db_client.get_collection_info(collection_name=collection_name)

        return collection_info

    async def index_into_vector_db(self, project:Project, chunks:List[DataChunk], 
                             chunks_ids:List[int], 
                             do_reset:bool = False):

        collection_name = self.create_collection_name(project_id=project.project_id)

        texts = [c.chunk_text for c in chunks]

        enhanced_metadata = []
        for c in chunks:
            meta = dict(c.chunk_metadata) if c.chunk_metadata else {}
            meta["asset_id"] = c.chunk_asset_id
            meta["chunk_order"] = c.chunk_order
            enhanced_metadata.append(meta)

        vectors = self.embedding_client.embed_text(
            text=texts, 
            document_type=DocumentTypeEnum.DOCUMENT.value, 
            input_type="passage"
        )
        batch_size = 50

        _ = await self.vector_db_client.create_collection(collection_name=collection_name,
                                                embedding_size = self.embedding_client.embedding_size,
                                                do_reset = do_reset                                     
                )

        _ = await self.vector_db_client.insert_many(
            collection_name=collection_name,
            texts=texts,
            vectors=vectors,
            metadata=enhanced_metadata,
            record_ids=chunks_ids,
            batch_size=batch_size
        )

        return True

    async def search_vector_db_collection(self, project:Project, text:str, limit: int=5, rerank_pool: int = 30):
        collection_name = self.create_collection_name(project_id=project.project_id)

        query_vector = None
        vectors = await asyncio.to_thread(
            self.embedding_client.embed_text,
            text=text, 
            document_type=DocumentTypeEnum.QUERY.value, 
            input_type="query"
        )
        if not vectors or len(vectors)==0 :
            return False

        if isinstance(vectors, list) and len(vectors)>0:
            query_vector = vectors[0]

        if not query_vector:
            return False

        candidates = await self.vector_db_client.search_by_vector(collection_name=collection_name,
                                                            vector=query_vector, k=rerank_pool
                                                )

        if not candidates:
            return False

        texts = [c.text for c in candidates]

        if self.reranker_client is not None:
            self.logger.info(f"starting rerank process")
            reranked = self.reranker_client.rerank(query=text, documents=texts, top_n=limit)    
            if not reranked:
                return candidates[:limit]
            return [candidates[i] for i, _ in reranked]

        return candidates[:limit]

    def detect_language(self, text: str) -> str:
        """Detect if text is primarily Arabic or English."""
        for char in text:
            if '\u0600' <= char <= '\u06FF':
                return 'ar'
        return 'en'

    async def answer_rag_query(self, project: Project, query: str, limit: int = 5, max_output_tokens: int = 1200):
        lang = self.detect_language(query)
        if self.template_parser:
            self.template_parser.set_language(lang)

        retrieved_docs = await self.search_vector_db_collection(project=project, text=query, limit=limit)

        if not retrieved_docs or len(retrieved_docs) == 0:
            refusal_msg = (
                "لا تتوفر لدي معلومات كافية في المستندات المقدمة للإجابة على هذا السؤال."
                if lang == 'ar'
                else "I do not have sufficient information in the provided documents to answer this question."
            )
            return refusal_msg, "", []

        doc_prompts = []
        for i, doc in enumerate(retrieved_docs):
            meta = doc.metadata or {}
            source_id = f"S{i+1}"
            source_name = meta.get("source_name", "unknown")

            if "page" in meta:
                location = f"Page {meta['page']}"
            elif "row" in meta:
                location = f"Row {meta['row']}"
            else:
                location = "N/A"

            doc_prompt = self.template_parser.get(
                "rag", "document_prompt",
                {
                    "source_id": source_id,
                    "source_name": source_name,
                    "location": location,
                    "chunk_text": doc.text
                }
            )
            doc_prompts.append(doc_prompt)

        document_context = "\n".join(doc_prompts)
        system_prompt = self.template_parser.get("rag", "system_prompt")

        voice_constraint = (
            " (قدم إجابة موجزة ومباشرة في جملتين إلى ثلاث جمل كحد أقصى)"
            if lang == "ar"
            else " (Provide a concise, direct answer in 2 to 3 sentences maximum)"
        )
        footer_prompt = self.template_parser.get("rag", "footer_prompt", {"query": f"{query}{voice_constraint}"})
        full_prompt = f"{document_context}\n\n{footer_prompt}"

        chat_history = [
            self.generation_client.construct_prompt(
                prompt=system_prompt,
                role=self.generation_client.enums.SYSTEM.value
            )
        ]

        answer = await asyncio.to_thread(
            self.generation_client.generate_text,
            prompt=full_prompt,
            chat_history=chat_history,
            max_output_token=max_output_tokens
        )

        return answer, full_prompt, chat_history

    def get_conversation_history(project:Project, conversation_id):
        pass

    def save_chat_turn(self, project_id:str, chat_history:list):
        all_collection =  self.db_client.list_collection_names()
        if DataBaseEnum.COLLECTION_CHUNK_NAME.value not in all_collection:
            self.collection = self.db_client[DataBaseEnum.COLLECTION_CHATS_NAME.value]
