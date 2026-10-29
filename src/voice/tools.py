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
        f"Retrieves grounded facts, regulations, and citations strictly bounded to {project_domain}."
    )

    @llm.function_tool(description=tool_desc)
    async def knowledge_base_rag(query: str) -> str:
        """Execute full RAG with document citations and honest refusals."""
        t0 = time.time()
        span = None
        if langfuse:
            span = langfuse.start_observation(
                name="knowledge_base_rag",
                as_type="span",
                input={"query": query, "project_id": project_id, "session_id": session_id},
                metadata={"project_id": project_id, "backend": "pgvector_rag"},
            )

        if not nlp_controller:
            msg = "NO_EVIDENCE_FOUND: NLP RAG service is not available."
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

            sources_cited = []
            if answer:
                citations = re.findall(r"\[S(\d+)\]", answer)
                if citations:
                    sources_cited = [f"S{c}" for c in sorted(set(citations))]

            if turn_context is not None:
                turn_context["last_rag_query"] = query
                turn_context["last_rag_answer"] = answer
                turn_context["last_sources_cited"] = sources_cited

            if span:
                span.update(
                    output={
                        "status": "answered" if answer else "insufficient_evidence",
                        "duration_ms": duration_ms,
                        "answer_preview": (answer or "")[:150],
                        "sources_cited": sources_cited,
                    }
                )
                span.end()

            if not answer or not answer.strip():
                return "NO_EVIDENCE_FOUND: No supporting facts found in the project's official documents."

            return f"Verified Project Answer:\n{answer}"

        except Exception as e:
            logger.error(f"Error in knowledge_base_rag tool: {e}")
            if span:
                span.update(output={"status": "error", "error": str(e)})
                span.end()
            return f"NO_EVIDENCE_FOUND: Error searching knowledge base: {e}"

    return knowledge_base_rag
