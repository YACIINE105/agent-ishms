import os
import json
import logging
from pathlib import Path

logger = logging.getLogger("uvicorn.error")

PERSONA_DIR = Path(__file__).resolve().parent.parent / "assets" / "personas"

def get_persona_file_path(project_id: int) -> Path:
    PERSONA_DIR.mkdir(parents=True, exist_ok=True)
    return PERSONA_DIR / f"project_{project_id}.json"

def load_project_persona(project_id: int) -> dict | None:
    """Load cached project persona from JSON disk storage."""
    file_path = get_persona_file_path(project_id)
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read persona file {file_path}: {e}")
    return None

def save_project_persona(project_id: int, persona_data: dict) -> bool:
    """Save generated project persona to JSON disk storage."""
    file_path = get_persona_file_path(project_id)
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(persona_data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"Failed to write persona file {file_path}: {e}")
        return False

def build_system_instructions(persona: dict) -> str:
    """Construct the agent's system prompt strictly bounded to this project's scope."""
    p_name = persona.get("name", "Project Assistant")
    p_domain = persona.get("domain", "Project knowledge and documents")
    p_summary = persona.get("summary", "Official project documentation and facts.")

    return (
        f"You are the official voice assistant for '{p_name}'.\n"
        f"Project Description: {p_summary}\n"
        f"Operational Boundaries: Your knowledge and answers are strictly limited to {p_domain}.\n\n"
        "CRITICAL LANGUAGE RULE:\n"
        "- ALWAYS answer in the EXACT SAME LANGUAGE as the user's spoken input.\n"
        "- If the user speaks in Arabic, you MUST speak in Arabic. NEVER translate an Arabic verified answer or question into English.\n"
        "- If the user speaks in English, you MUST speak in English.\n\n"
        "Core Operational Rules:\n"
        f"1. You ONLY assist with questions within the domain of '{p_name}'.\n"
        "2. For ANY question requiring facts, policies, or specific information, you MUST call "
        "the 'knowledge_base_rag' tool. CRITICAL: Pass the EXACT original user query to the tool in the user's language without translating, rephrasing into English/Chinese, or adding foreign words. Call the tool AT MOST ONCE per user turn.\n"
        "3. Once 'knowledge_base_rag' returns an answer, do NOT call any more tools. Immediately formulate your spoken response and speak to the user.\n"
        "4. Base your answer SOLELY on the verified RAG answer and state citations naturally (e.g. 'وفقاً للمرجع [S1]...' in Arabic, or 'According to [S1]...' in English).\n"
        "5. Keep the original language of the verified RAG answer. Do NOT translate Arabic facts into English.\n"
        "5. If the tool returns RAG_SERVICE_ERROR, the service has failed; briefly say the knowledge-base service is temporarily unavailable and ask the user to try again later, in their language. "
        "Do NOT claim the documents lack information, invent an answer, or speak the internal error marker. Apply this rule ONLY when the tool returns RAG_SERVICE_ERROR.\n"
        f"6. If 'knowledge_base_rag' returns no sufficient information, returns NO_EVIDENCE_FOUND, or if the question is outside '{p_domain}', "
        "you MUST give a strict, polite refusal of AT MOST TWO LINES (under 25 words) in the user's language. Never elaborate, guess, or apologize profusely.\n"
        "   Arabic refusal: 'عذراً، لا تتوفر معلومات كافية حول هذا الموضوع في لوائح الكلية الرسمية.'\n"
        "   English refusal: 'Sorry, I don't have sufficient information about that in the official project documents.'\n"
        "7. Keep all spoken answers concise (1-3 sentences) suitable for natural voice conversation.\n"
        "8. NEVER output internal XML tool tags like '<tool_call>' or ']<]minimax[>' in your spoken reply."
    )

async def generate_project_persona(
    project_id: int,
    sample_texts: list[str],
    generation_client=None
) -> dict:
    """
    Summarize embedded project chunks using the generation LLM, extracting
    name, domain, and operational boundaries, then persisting to assets/personas.
    """
    if not sample_texts:
        default_persona = {
            "project_id": project_id,
            "name": f"Project {project_id} Assistant",
            "domain": "general documentation and FAQs",
            "summary": "Official project documentation.",
        }
        save_project_persona(project_id, default_persona)
        return default_persona

    combined_sample = "\n---\n".join(sample_texts[:8])[:3000]

    prompt = (
        "You are an AI system architect. Based on the following document excerpts from a new project, "
        "determine the project's identity and operational boundaries.\n\n"
        f"Document Excerpts:\n{combined_sample}\n\n"
        "Output ONLY a valid JSON object with the following keys:\n"
        '{\n'
        '  "name": "Concise official name of the project / department / organization",\n'
        '  "domain": "Specific domain of topics covered (e.g. faculty regulations, tuition fees, registration)",\n'
        '  "summary": "1-2 sentence overview of what this project covers"\n'
        '}\n'
        "Do NOT include markdown formatting or backticks around the JSON."
    )

    try:
        if generation_client:
            raw_response = generation_client.generate_text(prompt=prompt)
            clean_json = raw_response.strip().replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean_json)
            persona_data = {
                "project_id": project_id,
                "name": parsed.get("name", f"Project {project_id}"),
                "domain": parsed.get("domain", "General documents"),
                "summary": parsed.get("summary", "Official project knowledge base."),
            }
        else:
            persona_data = {
                "project_id": project_id,
                "name": f"Project {project_id} Knowledge Base",
                "domain": "Project documents and FAQs",
                "summary": "Official project records and procedures.",
            }
    except Exception as e:
        logger.warning(f"LLM persona summarization failed, using default: {e}")
        persona_data = {
            "project_id": project_id,
            "name": f"Project {project_id} Assistant",
            "domain": "Project documents and regulations",
            "summary": "Official project records and procedures.",
        }

    save_project_persona(project_id, persona_data)
    logger.info(f"Generated & saved persona for project #{project_id}: '{persona_data['name']}'")
    return persona_data
