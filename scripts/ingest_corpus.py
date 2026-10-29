import asyncio
import os
import shutil
import sys
from pathlib import Path
from dotenv import load_dotenv

src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

root_dir = src_path.parent
load_dotenv(dotenv_path=root_dir / ".env.voice")
load_dotenv(dotenv_path=src_path / ".env")

from helpers.config import get_settings
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from models.db_schems import Project, Asset, DataChunk
from models.ProjectModel import ProjectModel
from models.AssetModel import AssetModel
from models.ChunkModel import ChunkModel
from models.enums.AssetTypeEnum import AssetTypeEnum
from controllers.ProcessController import ProcessController
from controllers.NLPController import NLPController
from stores.VectorDB import VectorDBPRoviderFactory
from stores.llm import LLMProviderFactory
from stores.llm.templates import TemplateParser

async def ingest():
    settings = get_settings()
    host = "localhost" if settings.POSTGRES_HOST == "pgvector" else settings.POSTGRES_HOST
    postgres_conn = (
        f"postgresql+asyncpg://{settings.POSTGRES_USERNAME}:{settings.POSTGRES_PASSWORD}@"
        f"{host}:{settings.POSTGRES_PORT}/{settings.POSTGRES_MAIN_DATABASE}"
    )
    db_engine = create_async_engine(postgres_conn)
    db_client = sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    print("--- 1. Setting up Project 1 ---")
    project_model = await ProjectModel.create_instance(db_client)
    project = await project_model.get_project_or_create_one(project_id=1)
    print(f"✓ Project ready: ID {project.project_id}")

    asset_model = await AssetModel.create_instance(db_client)
    chunk_model = await ChunkModel.create_instance(db_client)
    process_controller = ProcessController(project_id="1")

    llm_factory = LLMProviderFactory(settings)
    vector_db_factory = VectorDBPRoviderFactory(settings)

    vector_db_client = vector_db_factory.create(settings.VECTOR_DB_BACKEND)
    vector_db_client.db_client = db_client
    await vector_db_client.connect()

    embedding_client = llm_factory.create(provider=settings.EMBEDDING_PROVIDER)
    embedding_client.set_embedding_model(
        model_id=settings.EMBEDDING_MODEL_ID,
        embedding_size=settings.EMBEDDING_MODEL_SIZE,
    )

    template_parser = TemplateParser(language="en", default_language="en")
    nlp_controller = NLPController(
        generation_client=None,
        embedding_client=embedding_client,
        vector_db_client=vector_db_client,
        template_parser=template_parser,
    )

    project_files_dir = Path(src_path) / "assets" / "files" / "1"
    project_files_dir.mkdir(parents=True, exist_ok=True)

    corpus_files = [
        "student_services_faq.txt",
        "FEPS_student_handbook.pdf",
    ]

    for filename in corpus_files:
        src_file = root_dir / "data" / "corpus" / filename
        if not src_file.exists():
            print(f"Skipping {filename}: file not found at {src_file}")
            continue

        print(f"\n--- Ingesting {filename} ---")
        dest_file = project_files_dir / filename
        shutil.copy(src_file, dest_file)

        asset = Asset(
            asset_project_id=project.project_id,
            asset_type=AssetTypeEnum.FILE.value,
            asset_size=dest_file.stat().st_size,
            asset_name=filename,
            unique_asset_name=filename,
        )
        asset_record = await asset_model.create_asset(asset)
        print(f"✓ Created Asset record ID: {asset_record.asset_id}")

        file_content = process_controller.get_file_content(file_id=filename)
        chunks = process_controller.process_file_content(
            file_content=file_content,
            file_id=filename,
            chunk_size=400,
            overlap_size=50,
        )
        print(f"✓ Generated {len(chunks)} chunks")

        chunk_records = [
            DataChunk(
                chunk_project_id=project.project_id,
                chunk_asset_id=asset_record.asset_id,
                chunk_text=c.page_content,
                chunk_metadata=c.metadata,
                chunk_order=idx + 1,
            )
            for idx, c in enumerate(chunks)
        ]
        await chunk_model.delete_chunk_by_asset_id(asset_id=asset_record.asset_id)
        inserted = await chunk_model.insert_many_chunks(chunk_records)
        print(f"✓ Saved {inserted} chunks to PostgreSQL chunks table")

        print("✓ Computing embeddings and indexing into pgvector...")
        batch_size = 30
        collection_name = nlp_controller.create_collection_name(project.project_id)
        await vector_db_client.create_collection(
            collection_name=collection_name,
            embedding_size=settings.EMBEDDING_MODEL_SIZE,
            do_reset=False,
        )

        for b in range(0, len(chunk_records), batch_size):
            batch = chunk_records[b : b + batch_size]
            batch_texts = [c.chunk_text for c in batch]
            batch_ids = [c.chunk_id for c in batch]
            batch_meta = []
            for c in batch:
                m = dict(c.chunk_metadata) if c.chunk_metadata else {}
                m["asset_id"] = asset_record.asset_id
                m["chunk_order"] = c.chunk_order
                m["source_name"] = filename
                batch_meta.append(m)

            vectors = embedding_client.embed_text(
                text=batch_texts,
                document_type="document",
                input_type="passage",
            )
            await vector_db_client.insert_many(
                collection_name=collection_name,
                texts=batch_texts,
                vectors=vectors,
                metadata=batch_meta,
                record_ids=batch_ids,
                batch_size=batch_size,
            )
            print(f"  Indexed batch {b // batch_size + 1}/{(len(chunk_records) + batch_size - 1) // batch_size}")

        print(f"✓ Successfully indexed {filename} into pgvector!")

    print("\n--- Verifying Retrieval from pgvector ---")
    results = await nlp_controller.search_vector_db_collection(
        project=project,
        text="What is the student card fee?",
        limit=3,
    )
    print(f"Retrieved {len(results) if results else 0} documents:")
    if results:
        for i, r in enumerate(results):
            print(f"  [{i+1}] (Score: {r.score:.4f}, Asset: {r.asset_id}, Chunk: {r.chunk_id}): {r.text[:100]}...")

    await db_engine.dispose()
    print("\n✓ Ingestion & Vector Indexing Complete!")

if __name__ == "__main__":
    asyncio.run(ingest())
