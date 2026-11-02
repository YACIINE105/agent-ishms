"""Live 40-question evaluation of the production RAG path.

Every answer is produced by NLPController.answer_rag_query (the same code the
voice tool calls): pgvector retrieval -> 0.40 relevance gate -> grounded
generation with structured citations. Nothing is synthesized.
"""
import asyncio
import json
import re
import statistics
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

root_dir = Path(__file__).resolve().parent.parent
src_path = root_dir / "src"
sys.path = [str(root_dir), str(src_path)] + [p for p in sys.path if "4d-space/src" not in p]

load_dotenv(dotenv_path=src_path / ".env")
load_dotenv(dotenv_path=root_dir / ".env.voice")

from models.db_schems import Project  # noqa: E402

ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
TASHKEEL = re.compile(r"[\u064B-\u0652\u0640]")

REFUSAL_EN = [
    "not contain", "insufficient", "no information", "do not have", "unsupported",
    "cannot find", "not mentioned", "does not specify", "cannot be determined",
    "does not provide", "not found",
]
REFUSAL_AR = [
    "لا تحتوي", "غير متوفر", "لا توجد", "أعتذر", "لا تتوفر", "لا يذكر", "غير مذكور",
    "لا تتضمن", "لم يرد", "ليس لدي", "لا يمكن تحديد", "لا تتوافر", "لا تذكر", "لا يوجد",
]


def normalize(text: str) -> str:
    text = TASHKEEL.sub("", text.translate(ARABIC_DIGITS)).lower()
    text = text.replace("٪", "%").replace("*", "").replace("\u202f", " ").replace("\u00a0", " ")
    text = re.sub(r"[إأآ]", "ا", text).replace("ى", "ي").replace("ة", "ه")
    text = re.sub(r"(?<=\d)[, ](?=\d{3}\b)", "", text)
    text = re.sub(r"(?<=\d)\s+%", "%", text)
    return re.sub(r"\s+", " ", text).strip()

def check_retrieval_hit(expected_locs, retrieved_docs):
    """Rank (1-based) of the first retrieved chunk from an expected source, else 0."""
    for rank, doc in enumerate(retrieved_docs, start=1):
        src = str((doc.metadata or {}).get("source_name", "")).lower()
        for exp in expected_locs:
            exp_src = exp.get("source_name", "").lower()
            if exp_src and (exp_src in src or src in exp_src):
                return rank
    return 0


def check_refusal(answer: str, is_arabic: bool) -> bool:
    if is_arabic:
        return any(k in answer for k in REFUSAL_AR)
    lowered = answer.lower()
    return any(k in lowered for k in REFUSAL_EN)


def fact_recall(answer: str, facts: list[str]) -> float:
    if not facts:
        return 1.0
    norm_answer = normalize(answer)
    return sum(1 for f in facts if normalize(f) in norm_answer) / len(facts)


def pct(n: int, d: int) -> float:
    return round(100 * n / d, 1) if d else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))], 1)


async def answer_with_retry(nlp_controller, project, question, attempts=4):
    """Groq free tier is 8k TPM; provider returns None on 429, so back off and retry."""
    for attempt in range(attempts):
        answer, _, _ = await nlp_controller.answer_rag_query(project=project, query=question, limit=5)
        if answer:
            return answer
        await asyncio.sleep(8 * (attempt + 1))
    return ""


async def run_evaluation():
    from eval.compare_retrieval_40 import build_isolated_nlp_controller
    from helpers.config import get_settings

    settings = get_settings()
    nlp_controller = build_isolated_nlp_controller()
    project = Project(project_id=1)
    model_name = f"{settings.GENERATION_PROVIDER}:{settings.GENERATION_MODEL_ID}"

    questions_path = root_dir / "eval" / "questions_40.jsonl"
    with open(questions_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(records)} questions | generation model: {model_name}")

    results = []
    for i, r in enumerate(records, start=1):
        is_ar = r["language"] == "ar"
        answerable = r["answerable"]

        t0 = time.perf_counter()
        docs = await nlp_controller.search_vector_db_collection(project=project, text=r["question"], limit=5) or []
        retrieval_ms = (time.perf_counter() - t0) * 1000
        rank = check_retrieval_hit(r["expected_source_locations"], docs) if answerable else 0
        top_score = max((getattr(d, "score", 0.0) or 0.0 for d in docs), default=0.0)

        t1 = time.perf_counter()
        answer = await answer_with_retry(nlp_controller, project, r["question"])
        rag_ms = (time.perf_counter() - t1) * 1000
        citations = list(nlp_controller.last_citations or [])

        refused = check_refusal(answer, is_ar)
        recall = fact_recall(answer, r["reference_facts"]) if answerable else None
        has_citation = bool(citations) and bool(re.search(r"[\[【]S\d+[\]】]", answer))

        results.append({
            "id": r["id"],
            "language": r["language"],
            "answerable": answerable,
            "question": r["question"],
            "hit_rank": rank,
            "top_score": round(top_score, 4),
            "retrieval_ms": round(retrieval_ms, 1),
            "rag_end_to_end_ms": round(rag_ms, 1),
            "refused": refused,
            "fact_recall": recall,
            "has_citation": has_citation,
            "citations": citations,
            "generated_answer": answer,
        })
        status = f"rank={rank} recall={recall}" if answerable else f"refused={refused}"
        print(f"[{i:02d}/{len(records)}] {r['id']} {r['language']} {status} "
              f"cite={has_citation} ret={retrieval_ms:.0f}ms rag={rag_ms:.0f}ms")

    ans = [x for x in results if x["answerable"]]
    ood = [x for x in results if not x["answerable"]]
    ranks = [x["hit_rank"] for x in ans]

    summary = {
        "total_questions": len(results),
        "answerable": len(ans),
        "unanswerable": len(ood),
        "generation_model": model_name,
        "embedding_model": settings.EMBEDDING_MODEL_ID,
        "hit_at_1_pct": pct(sum(1 for k in ranks if k == 1), len(ans)),
        "hit_at_3_pct": pct(sum(1 for k in ranks if 0 < k <= 3), len(ans)),
        "hit_at_5_pct": pct(sum(1 for k in ranks if 0 < k <= 5), len(ans)),
        "mrr": round(sum(1 / k for k in ranks if k) / len(ans), 3) if ans else 0.0,
        "mean_fact_recall_pct": round(100 * statistics.mean(x["fact_recall"] for x in ans), 1) if ans else 0.0,
        "exact_fact_match_pct": pct(sum(1 for x in ans if x["fact_recall"] == 1.0), len(ans)),
        "citation_rate_pct": pct(sum(1 for x in ans if x["has_citation"]), len(ans)),
        "false_refusal_pct": pct(sum(1 for x in ans if x["refused"]), len(ans)),
        "ood_refusal_accuracy_pct": pct(sum(1 for x in ood if x["refused"]), len(ood)),
        "retrieval_ms_p50": percentile([x["retrieval_ms"] for x in results], 0.5),
        "retrieval_ms_p95": percentile([x["retrieval_ms"] for x in results], 0.95),
        "rag_end_to_end_ms_p50": percentile([x["rag_end_to_end_ms"] for x in results], 0.5),
        "rag_end_to_end_ms_p95": percentile([x["rag_end_to_end_ms"] for x in results], 0.95),
    }
    for lang in ("en", "ar"):
        sub = [x for x in ans if x["language"] == lang]
        summary[f"{lang}_mean_fact_recall_pct"] = round(100 * statistics.mean(x["fact_recall"] for x in sub), 1) if sub else 0.0

    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))
    results_path = root_dir / "eval" / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_question_results": results}, f, ensure_ascii=False, indent=2)
    print(f"Saved {results_path}")


if __name__ == "__main__":
    asyncio.run(run_evaluation())
