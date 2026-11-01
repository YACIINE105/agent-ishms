import time
import re
import logging
from livekit.agents import llm

logger = logging.getLogger("livekit-agent")

def create_knowledge_base_rag_tool(
    project_id: int,
    project_name: str,
    project_domain: str,
    nlp_controller,
    langfuse=None,
    session_id: str = "",
    turn_context: dict | None = None,
):
    """
    Factory creating a LiveKit function tool that invokes the project's full
    RAG pipeline (NLPController.answer_rag_query) with citation support.
    """
    tool_desc = (
        f"Query the official verified knowledge base for '{project_name}'. "
        f"Retrieves grounded facts, regulations, and citations strictly bounded to {project_domain}. "
        "Pass the user's natural query directly without rephrasing or translating into other languages."
    )
    @llm.function_tool(description=tool_desc)
    async def knowledge_base_rag(query: str) -> str:
        """Execute full RAG with document citations and honest refusals."""
        t0 = time.time()
        if turn_context is not None:
            turn_context.update(last_rag_query=query, last_rag_answer=None, last_sources_cited=[])
        span = None
        if langfuse:
            span = langfuse.start_observation(
                name="knowledge_base_rag",
                as_type="span",
                input={"query": query, "project_id": project_id, "session_id": session_id},
                metadata={"project_id": project_id, "backend": "pgvector_rag"},
            )

        if not nlp_controller:
            msg = "RAG_SERVICE_ERROR: The knowledge-base service is temporarily unavailable. Please try again later."
            if span:
                span.update(output={"status": "error", "message": msg})
                span.end()
            return msg

        try:
            from models.db_schems import Project
            proj = Project(project_id=project_id)

            answer, full_prompt, _ = await nlp_controller.answer_rag_query(
                project=proj,
                query=query,
                limit=5,
            )
            duration_ms = round((time.time() - t0) * 1000, 2)
            if not answer or not answer.strip():
                logger.error("RAG generation returned no answer")
                if span:
                    span.update(output={"status": "error", "error": "generation_failed", "duration_ms": duration_ms})
                    span.end()
                return "RAG_SERVICE_ERROR: The knowledge-base answer service is temporarily unavailable. Please try again later."

            sources_cited = []
            if answer:
                citations = re.findall(r"\[S(\d+)\]", answer)
                if citations:
                    sources_cited = [f"S{c}" for c in sorted(set(citations))]

            if turn_context is not None:
                turn_context["last_rag_query"] = query
                turn_context["last_rag_answer"] = answer
                turn_context["last_sources_cited"] = sources_cited

            is_abstention = (not full_prompt) or ("لا تتوفر لدي معلومات" in (answer or "")) or ("do not have sufficient information" in (answer or "").lower())
            abstention_reason = "score_below_threshold" if is_abstention else None

            if span:
                output_meta = {
                    "status": "answered" if not is_abstention else "abstention_refusal",
                    "duration_ms": duration_ms,
                    "answer_preview": (answer or "")[:150],
                    "sources_cited": sources_cited,
                    "abstention_triggered": is_abstention,
                }
                if abstention_reason:
                    output_meta["abstention_reason"] = abstention_reason
                span.update(output=output_meta)
                span.end()

            if is_abstention:
                return f"NO_EVIDENCE_FOUND: {answer}"
            return f"Verified Project Answer:\n{answer}"

        except Exception as e:
            logger.error("RAG service failed (%s)", type(e).__name__)
            if span:
                span.update(output={"status": "error", "error": type(e).__name__})
                span.end()
            return "RAG_SERVICE_ERROR: The knowledge-base service is temporarily unavailable. Please try again later."

    return knowledge_base_rag
