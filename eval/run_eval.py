import asyncio
import json
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

root_dir = Path(__file__).resolve().parent.parent
src_path = root_dir / "src"
sys.path.insert(0, str(src_path))

load_dotenv(dotenv_path=src_path / ".env")
load_dotenv(dotenv_path=root_dir / ".env.voice")

import openai
from voice.agent import build_rag_controller
from models.db_schems import Project

def get_llm_client():
    """Get active generation client with fallback (A2Agent -> Google -> OpenRouter)."""
    a2_key = os.getenv("A2AGENT_API_KEY")
    if a2_key:
        return openai.OpenAI(
            base_url=os.getenv("A2AGENT_BASE_URL", "https://api.a2agent.me/v1").strip("\"'"),
            api_key=a2_key.strip("\"'"),
        ), os.getenv("A2AGENT_MODEL_ID", "minimax-m3.1-flash")

    g_key = os.getenv("GOOGLE_AI_API_KEY")
    if g_key:
        return openai.OpenAI(
            base_url=os.getenv("GOOGLE_AI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/").strip("\"'"),
            api_key=g_key.strip("\"'"),
        ), os.getenv("GOOGLE_AI_MODEL_ID", "gemini-2.5-flash")

    or_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("openrouter_api_key")
    if or_key and or_key.startswith("["):
        or_key = json.loads(or_key)[0]
    return openai.OpenAI(
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip("\"'"),
        api_key=or_key.strip("\"'"),
    ), os.getenv("GENERATION_MODEL_ID", "google/gemini-2.5-flash")

def check_retrieval_hit(expected_locs, retrieved_docs):
    """Check if any of the expected sources are in the retrieved top-k."""
    if not expected_locs:
        return True, 0

    first_hit_rank = None
    for rank, doc in enumerate(retrieved_docs, start=1):
        meta = doc.metadata or {}
        src = meta.get("source_name", "")
        for exp in expected_locs:
            exp_src = exp.get("source_name", "")
            if exp_src in src or src in exp_src:
                if first_hit_rank is None:
                    first_hit_rank = rank
                break

    hit = first_hit_rank is not None
    return hit, first_hit_rank or 0

def check_refusal(answer: str, is_arabic: bool) -> bool:
    """Detect if the answer is an honest refusal."""
    ans_lower = answer.lower()
    refusal_keywords_en = [
        "not contain", "insufficient", "no information", "do not have", 
        "unsupported", "cannot find", "not mentioned", "apologize", "sorry", 
        "does not specify", "cannot be determined", "does not provide", "not found"
    ]
    refusal_keywords_ar = [
        "لا تحتوي", "غير متوفر", "لا توجد", "أعتذر", "لا تتوفر", 
        "لا يذكر", "غير مذكور", "لا تتضمن", "لم يرد", "ليس لدي",
        "لا تتضمّن", "لا يمكن تحديد", "لا تتوافر", "لا تذكر", "لا يوجد"
    ]
    keywords = refusal_keywords_ar if is_arabic else refusal_keywords_en
    return any(k in (answer if is_arabic else ans_lower) for k in keywords)

def check_citations(answer: str) -> bool:
    """Check if the answer cites sources (e.g., [S1], Page, Section, Guide, Handbook)."""
    ans_lower = answer.lower()
    indicators = [
        "[s1]", "[s2]", "[s3]", "page", "section", "table", "row", 
        "handbook", "guide", "faq", "وفقاً", "حسب", "القسم", "صفحة", "جدول", "الدليل"
    ]
    return any(ind in ans_lower or ind in answer for ind in indicators)

async def run_evaluation():
    print("=" * 70)
    print("Starting Automated 20-Question RAG Evaluation Suite")
    print("=" * 70)

    nlp_controller = build_rag_controller()
    project = Project(project_id=1)
    client, model_name = get_llm_client()
    print(f"Using LLM for generation: {model_name}")

    questions_path = root_dir / "eval" / "questions.jsonl"
    with open(questions_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    results = []
    retrieval_hits_at_1 = 0
    retrieval_hits_at_3 = 0
    retrieval_hits_at_5 = 0
    reciprocal_ranks = []
    grounded_answers = 0
    valid_citations = 0
    correct_refusals = 0
    total_unanswerable = 0
    total_answerable = 0

    for i, r in enumerate(records, start=1):
        qid = r["id"]
        q_text = r["question"]
        lang = r["language"]
        is_ar = lang == "ar"
        answerable = r["answerable"]
        exp_locs = r["expected_source_locations"]
        ref_facts = r["reference_facts"]
        history = r.get("history", [])

        print(f"\n[{i}/20] Evaluating {qid} ({lang}): {q_text[:60]}...")
        t0 = time.time()

        docs = await nlp_controller.search_vector_db_collection(project=project, text=q_text, limit=5)
        ret_time = round(time.time() - t0, 3)

        hit, rank = check_retrieval_hit(exp_locs, docs if docs else [])

        if answerable:
            total_answerable += 1
            if hit:
                if rank == 1:
                    retrieval_hits_at_1 += 1
                if rank <= 3:
                    retrieval_hits_at_3 += 1
                if rank <= 5:
                    retrieval_hits_at_5 += 1
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)
        else:
            total_unanswerable += 1

        context_parts = []
        if docs:
            for idx, d in enumerate(docs[:3], start=1):
                meta = d.metadata or {}
                src = meta.get("source_name", "Handbook")
                loc = f"Page {meta.get('page')}" if "page" in meta else f"Row {meta.get('row', 'N/A')}"
                context_parts.append(f"[S{idx}] ({src}, {loc}): {d.text}")
        context_str = "\n\n".join(context_parts) if context_parts else "NO_EVIDENCE_FOUND"

        system_prompt = (
            "You are Gareth, an official university student services assistant.\n"
            "Rules:\n"
            "1. Answer based ONLY on the provided context evidence.\n"
            "2. State your sources naturally (e.g. 'According to Section 6.1...' or '[S1]').\n"
            "3. If the context does not contain sufficient facts to answer the question, refuse politely.\n"
            "4. Reply in fluent " + ("Arabic." if is_ar else "English.")
        )

        messages = [{"role": "system", "content": system_prompt}]
        for h in history:
            messages.append(h)
        messages.append({
            "role": "user",
            "content": f"Context Evidence:\n{context_str}\n\nQuestion: {q_text}"
        })

        t_gen_start = time.time()
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=600,
                temperature=0.1,
            )
            answer_text = resp.choices[0].message.content or ""
        except Exception as e:
            answer_text = f"GENERATION_ERROR: {e}"
        gen_time = round(time.time() - t_gen_start, 3)

        has_cite = check_citations(answer_text)
        is_refusal = check_refusal(answer_text, is_ar)

        if not answerable:
            refusal_correct = is_refusal
            if refusal_correct:
                correct_refusals += 1
            grounded = refusal_correct
        else:
            refusal_correct = False
            grounded = has_cite and not is_refusal
            if grounded:
                grounded_answers += 1
            if has_cite:
                valid_citations += 1

        print(f"  ✓ Ret: {ret_time}s (Hit@{rank if rank else 'None'}), Gen: {gen_time}s")
        print(f"  ✓ Citations: {has_cite}, Refusal: {is_refusal}, Grounded: {grounded}")
        print(f"  Spoken Output: {answer_text[:120]}...")

        results.append({
            "id": qid,
            "language": lang,
            "answerable": answerable,
            "question": q_text,
            "retrieval_hit": hit,
            "hit_rank": rank,
            "retrieval_time_s": ret_time,
            "generation_time_s": gen_time,
            "has_citations": has_cite,
            "is_refusal": is_refusal,
            "grounded": grounded,
            "generated_answer": answer_text,
        })

    hit_at_1 = round(retrieval_hits_at_1 / total_answerable * 100, 1)
    hit_at_3 = round(retrieval_hits_at_3 / total_answerable * 100, 1)
    hit_at_5 = round(retrieval_hits_at_5 / total_answerable * 100, 1)
    mrr = round(sum(reciprocal_ranks) / total_answerable, 3)
    refusal_acc = round(correct_refusals / total_unanswerable * 100, 1)
    groundedness = round(grounded_answers / total_answerable * 100, 1)
    citation_rate = round(valid_citations / total_answerable * 100, 1)

    print("\n" + "=" * 70)
    print("FINAL EVALUATION METRICS (20 Questions)")
    print("=" * 70)
    print(f"Hit Rate @ 1:         {hit_at_1}% ({retrieval_hits_at_1}/{total_answerable})")
    print(f"Hit Rate @ 3:         {hit_at_3}% ({retrieval_hits_at_3}/{total_answerable})")
    print(f"Hit Rate @ 5:         {hit_at_5}% ({retrieval_hits_at_5}/{total_answerable})")
    print(f"Mean Reciprocal Rank: {mrr}")
    print(f"Refusal Accuracy:     {refusal_acc}% ({correct_refusals}/{total_unanswerable})")
    print(f"Answer Groundedness:  {groundedness}% ({grounded_answers}/{total_answerable})")
    print(f"Citation Rate:        {citation_rate}% ({valid_citations}/{total_answerable})")
    print("=" * 70)

    eval_out = {
        "summary": {
            "total_questions": len(records),
            "answerable": total_answerable,
            "unanswerable": total_unanswerable,
            "model": model_name,
            "hit_at_1_pct": hit_at_1,
            "hit_at_3_pct": hit_at_3,
            "hit_at_5_pct": hit_at_5,
            "mrr": mrr,
            "refusal_accuracy_pct": refusal_acc,
            "groundedness_pct": groundedness,
            "citation_rate_pct": citation_rate,
        },
        "per_question_results": results,
    }

    results_path = root_dir / "eval" / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(eval_out, f, ensure_ascii=False, indent=2)
    print(f"Detailed results saved to {results_path}")

if __name__ == "__main__":
    asyncio.run(run_evaluation())
