import asyncio
import os
import shutil
import sys
from pathlib import Path
from dotenv import load_dotenv

src_path = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_path))
root_dir = src_path.parent

load_dotenv(dotenv_path=src_path / ".env")
load_dotenv(dotenv_path=root_dir / ".env.voice")

from helpers.config import get_settings
from models.db_schems import Project, Asset, DataChunk
from models.AssetModel import AssetModel
from models.ChunkModel import ChunkModel
from models.enums.AssetTypeEnum import AssetTypeEnum
from controllers import ProcessController, NLPController
from stores.llm import LLMProviderFactory
from stores.VectorDB import VectorDBPRoviderFactory
from stores.llm.templates import TemplateParser
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

async def ingest_csv():
    settings = get_settings()
    host = "localhost" if settings.POSTGRES_HOST == "pgvector" else settings.POSTGRES_HOST
    postgres_conn = (
        f"postgresql+asyncpg://{settings.POSTGRES_USERNAME}:{settings.POSTGRES_PASSWORD}@"
        f"{host}:{settings.POSTGRES_PORT}/{settings.POSTGRES_MAIN_DATABASE}"
    )
    db_engine = create_async_engine(postgres_conn)
    db_client = sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    asset_model = AssetModel(db_client)
    chunk_model = ChunkModel(db_client)
    process_controller = ProcessController(project_id="1")

    llm_factory = LLMProviderFactory(settings)
    vector_db_factory = VectorDBPRoviderFactory(settings)

    vector_db_client = vector_db_factory.create(settings.VECTOR_DB_BACKEND)
    vector_db_client.db_client = db_client

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

    project = Project(project_id=1)
    project_files_dir = Path(src_path) / "assets" / "files" / "1"
    project_files_dir.mkdir(parents=True, exist_ok=True)

    csv_src = root_dir / "data" / "corpus" / "diversity_school.csv"
    dest_file = project_files_dir / "diversity_school.csv"

    with open(csv_src, "r", encoding="utf-8") as f:
        lines = [f.readline() for _ in range(55)]
    with open(dest_file, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"Extracted {len(lines)} lines from diversity_school.csv")

    asset = Asset(
        asset_project_id=project.project_id,
        asset_type=AssetTypeEnum.FILE.value,
        asset_size=dest_file.stat().st_size,
        asset_name="diversity_school.csv",
        unique_asset_name="diversity_school.csv",
    )
    asset_record = await asset_model.create_asset(asset)
    print(f"✓ Created Asset record ID: {asset_record.asset_id}")

    file_content = process_controller.get_file_content(file_id="diversity_school.csv")
    chunks = process_controller.process_file_content(
        file_content=file_content,
        file_id="diversity_school.csv",
        chunk_size=400,
        overlap_size=50,
    )
    print(f"✓ Generated {len(chunks)} CSV chunks")

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
    batch_size = 25
    collection_name = nlp_controller.create_collection_name(project.project_id)

    for b in range(0, len(chunk_records), batch_size):
        batch = chunk_records[b : b + batch_size]
        texts = [c.chunk_text for c in batch]
        record_ids = [c.chunk_id for c in batch]

        vectors = embedding_client.embed_text(
            text=texts,
            document_type="document",
            input_type="passage",
        )
        enhanced_meta = []
        for c in batch:
            m = dict(c.chunk_metadata) if c.chunk_metadata else {}
            m["asset_id"] = asset_record.asset_id
            m["chunk_order"] = c.chunk_order
            m["source_name"] = "diversity_school.csv"
            enhanced_meta.append(m)

        await vector_db_client.insert_many(
            collection_name=collection_name,
            texts=texts,
            vectors=vectors,
            metadata=enhanced_meta,
            record_ids=record_ids,
            batch_size=batch_size,
        )
        print(f"  Indexed batch {b // batch_size + 1} ({len(batch)} chunks)")

    print("✓ CSV Ingestion complete!")
    await db_engine.dispose()

if __name__ == "__main__":
    asyncio.run(ingest_csv())
