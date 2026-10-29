import sys
import asyncio
from pathlib import Path

src_path = Path(__file__).resolve().parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from controllers.NLPController import NLPController
from stores.llm.templates import TemplateParser
from models.db_schems import Project, RetrievedDocuments

controller = NLPController.__new__(NLPController)
controller.template_parser = TemplateParser(default_language='en')

assert controller.detect_language("How much is the student fee?") == "en"
assert controller.detect_language("كم هي رسوم استخراج بطاقة الطالب؟") == "ar"
print("✓ Language detection works for both Arabic and English!")

controller.template_parser.set_language("en")
sys_en = controller.template_parser.get("rag", "system_prompt")
assert "strictly grounded" in sys_en
assert "[S1]" in sys_en
print("✓ English grounded prompt and citation format verified!")

controller.template_parser.set_language("ar")
sys_ar = controller.template_parser.get("rag", "system_prompt")
assert "مساعد معلوماتي دقيق" in sys_ar
assert "[S1]" in sys_ar
print("✓ Arabic grounded prompt and citation format verified!")

async def mock_zero_search(project, text, limit):
    return []

controller.search_vector_db_collection = mock_zero_search

async def run_refusal_tests():
    ans_en, prompt_en, history_en = await controller.answer_rag_query(
        Project(), "What is the fee for veterinary registration?"
    )
    assert "not have sufficient information" in ans_en
    print(f"✓ English refusal verified: '{ans_en}'")

    ans_ar, prompt_ar, history_ar = await controller.answer_rag_query(
        Project(), "ما هي رسوم التسجيل لكلية الطب البيطري؟"
    )
    assert "لا تتوفر لدي معلومات كافية" in ans_ar
    print(f"✓ Arabic refusal verified: '{ans_ar}'")

asyncio.run(run_refusal_tests())

mock_docs = [
    RetrievedDocuments(
        text="Student card replacement fee is 500 DZD.",
        score=0.95,
        chunk_id=1,
        asset_id=1,
        metadata={"source_name": "handbook.pdf", "page": 3}
    ),
    RetrievedDocuments(
        text="Tuition for semester: 1200 DZD.",
        score=0.88,
        chunk_id=2,
        asset_id=2,
        metadata={"source_name": "fees.csv", "row": 15}
    )
]

controller.template_parser.set_language("en")
doc_prompt_1 = controller.template_parser.get(
    "rag", "document_prompt",
    {"source_id": "S1", "source_name": "handbook.pdf", "location": "Page 3", "chunk_text": mock_docs[0].text}
)
assert "[S1]" in doc_prompt_1
assert "Page 3" in doc_prompt_1
print("✓ Source context formatting with [S1] and location verified!")

print("\nAll bilingual retrieval and refusal tests passed successfully!")
