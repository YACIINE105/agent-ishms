import asyncio
import json
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

root_dir = Path(__file__).resolve().parent.parent
src_path = root_dir / "src"
sys.path = [str(root_dir), str(src_path)] + [p for p in sys.path if "4d-space/src" not in p]

load_dotenv(dotenv_path=src_path / ".env")
load_dotenv(dotenv_path=root_dir / ".env.voice")

from models.db_schems import Project
from stores.llm.rerankers.cohere_reranker import CohereReranker


def check_retrieval_hit(expected_locs, retrieved_docs):
    if not expected_locs:
        return True, 0
    first_hit_rank = None
    for rank, doc in enumerate(retrieved_docs, start=1):
        meta = doc.metadata or {}
        src = meta.get("source_name", "") or meta.get("source_document", "")
        for exp in expected_locs:
            exp_src = exp.get("source_name", "")
            if exp_src.lower() in src.lower() or src.lower() in exp_src.lower():
                if first_hit_rank is None:
                    first_hit_rank = rank
                break
    hit = first_hit_rank is not None
    return hit, first_hit_rank or 0


def build_isolated_nlp_controller():
    from controllers.NLPController import NLPController
    from helpers.config import get_settings
    from stores.VectorDB import VectorDBPRoviderFactory
    from stores.llm import LLMProviderFactory
    from stores.llm.templates import TemplateParser
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker

    settings = get_settings()
    host = "localhost" if settings.POSTGRES_HOST == "pgvector" else settings.POSTGRES_HOST
    postgres_conn = (
        f"postgresql+asyncpg://{settings.POSTGRES_USERNAME}:{settings.POSTGRES_PASSWORD}@"
        f"{host}:{settings.POSTGRES_PORT}/{settings.POSTGRES_MAIN_DATABASE}"
    )
    db_engine = create_async_engine(postgres_conn)
    db_client = sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    llm_factory = LLMProviderFactory(settings)
    vector_db_factory = VectorDBPRoviderFactory(settings)

    vector_db_client = vector_db_factory.create(settings.VECTOR_DB_BACKEND)
    vector_db_client.db_client = db_client

    embedding_client = llm_factory.create(provider=settings.EMBEDDING_PROVIDER, for_embeddings=True)
    embedding_client.set_embedding_model(
        model_id=settings.EMBEDDING_MODEL_ID,
        embedding_size=settings.EMBEDDING_MODEL_SIZE,
    )

    generation_client = llm_factory.create(provider=settings.GENERATION_PROVIDER)
    generation_client.set_generation_model(model_id=settings.GENERATION_MODEL_ID)

    template_parser = TemplateParser(language=settings.PRIMARY_LANG, default_language=settings.DEFAULT_LANG)

    return NLPController(
        generation_client=generation_client,
        embedding_client=embedding_client,
        vector_db_client=vector_db_client,
        reranker_client=None,
        template_parser=template_parser,
    )


async def evaluate_retrieval_comparison():
    nlp_controller = build_isolated_nlp_controller()
    project = Project(project_id=1)
    cohere_key = os.getenv("COHERE_API_KEY")

    questions_path = root_dir / "eval" / "questions_40.jsonl"
    with open(questions_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(records)} benchmark questions for Retrieval & Reranker Comparison.\n")

    # Pass 1: Pure Dense Vector Search (Without Reranker)
    print(">>> [Pass 1/2] Evaluating Pure Dense Vector Search (Without Reranker)...", flush=True)
    nlp_controller.reranker_client = None
    dense_results = []
    dense_latencies = []

    for idx, r in enumerate(records, start=1):
        q = r["question"]
        t0 = time.time()
        docs = await nlp_controller.search_vector_db_collection(project=project, text=q, limit=5)
        dt = time.time() - t0
        dense_latencies.append(dt)
        hit, rank = check_retrieval_hit(r["expected_source_locations"], docs or [])
        dense_results.append({
            "id": r["id"], "answerable": r["answerable"], "hit": hit, "rank": rank, "latency": dt
        })

    # Pass 2: With Remote Cohere Reranker (Sample first 10 due to 10 RPM Trial Limit)
    print("\n>>> [Pass 2/2] Evaluating With Remote Cohere Reranker (Sampled to avoid 10 RPM Trial Rate Limit)...", flush=True)
    nlp_controller.reranker_client = CohereReranker(api_key=cohere_key, model="rerank-v3.5")
    rerank_results = []
    rerank_latencies = []

    sample_records = records[:10]  # Respect 10 calls/min rate limit
    for idx, r in enumerate(sample_records, start=1):
        q = r["question"]
        t0 = time.time()
        docs = await nlp_controller.search_vector_db_collection(project=project, text=q, limit=5)
        dt = time.time() - t0
        rerank_latencies.append(dt)
        hit, rank = check_retrieval_hit(r["expected_source_locations"], docs or [])
        rerank_results.append({
            "id": r["id"], "answerable": r["answerable"], "hit": hit, "rank": rank, "latency": dt
        })
        time.sleep(1.0)  # gentle pacing

    # Metrics computation
    ans_dense = [r for r in dense_results if r["answerable"]]
    hit1_dense = sum(1 for r in ans_dense if r["rank"] == 1) / len(ans_dense)
    hit3_dense = sum(1 for r in ans_dense if 1 <= r["rank"] <= 3) / len(ans_dense)
    hit5_dense = sum(1 for r in ans_dense if 1 <= r["rank"] <= 5) / len(ans_dense)
    mrr_dense = sum(1.0/r["rank"] for r in ans_dense if r["rank"] > 0) / len(ans_dense)
    avg_lat_dense = (sum(dense_latencies) / len(dense_latencies)) * 1000

    ans_rerank = [r for r in rerank_results if r["answerable"]]
    hit1_rerank = sum(1 for r in ans_rerank if r["rank"] == 1) / len(ans_rerank)
    hit3_rerank = sum(1 for r in ans_rerank if 1 <= r["rank"] <= 3) / len(ans_rerank)
    hit5_rerank = sum(1 for r in ans_rerank if 1 <= r["rank"] <= 5) / len(ans_rerank)
    mrr_rerank = sum(1.0/r["rank"] for r in ans_rerank if r["rank"] > 0) / len(ans_rerank)
    avg_lat_rerank = (sum(rerank_latencies) / len(rerank_latencies)) * 1000

    print("\n" + "=" * 80)
    print("40-QUESTION RETRIEVAL & RERANKER COMPARISON RESULTS")
    print("=" * 80)
    print(f"{'Metric':<35} | {'Without Reranker':<18} | {'With Cohere Reranker':<20} | {'Delta':<10}")
    print("-" * 88)
    print(f"{'Mean Reciprocal Rank (MRR)':<35} | {mrr_dense*100:>16.1f}% | {mrr_rerank*100:>18.1f}% | {(mrr_rerank-mrr_dense)*100:>+8.1f}%")
    print(f"{'Hit Rate @ 1 (Top-1 Recall)':<35} | {hit1_dense*100:>16.1f}% | {hit1_rerank*100:>18.1f}% | {(hit1_rerank-hit1_dense)*100:>+8.1f}%")
    print(f"{'Hit Rate @ 3 (Top-3 Recall)':<35} | {hit3_dense*100:>16.1f}% | {hit3_rerank*100:>18.1f}% | {(hit3_rerank-hit3_dense)*100:>+8.1f}%")
    print(f"{'Hit Rate @ 5 (Top-5 Recall)':<35} | {hit5_dense*100:>16.1f}% | {hit5_rerank*100:>18.1f}% | {(hit5_rerank-hit5_dense)*100:>+8.1f}%")
    print(f"{'Avg Retrieval Latency':<35} | {avg_lat_dense:>15.1f} ms | {avg_lat_rerank:>17.1f} ms | {avg_lat_rerank-avg_lat_dense:>+8.1f} ms")
    print("=" * 88)

    return {
        "mrr_dense": mrr_dense, "hit1_dense": hit1_dense, "hit3_dense": hit3_dense, "hit5_dense": hit5_dense, "avg_lat_dense": avg_lat_dense,
        "mrr_rerank": mrr_rerank, "hit1_rerank": hit1_rerank, "hit3_rerank": hit3_rerank, "hit5_rerank": hit5_rerank, "avg_lat_rerank": avg_lat_rerank
    }


if __name__ == "__main__":
    asyncio.run(evaluate_retrieval_comparison())
