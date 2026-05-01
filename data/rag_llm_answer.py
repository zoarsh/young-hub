# ===============================
# כותרת: ניסוח תשובה בעזרת OpenAI API
# הסבר: מודול זה מקבל את תוצאות השליפה של ה-RAG ומנסח תשובה עברית
#       רק על בסיס קטעי המקור שנשלפו, עם ציטוטי עמוד ברורים.
# ===============================

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


# ===============================
# שלב: טעינת משתני סביבה
# הסבר: טוען את OPENAI_API_KEY מתוך קובץ .env מקומי, בלי לשמור מפתח בקוד.
# ===============================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


# ===============================
# שלב: הגדרת מודל ברירת מחדל
# הסבר: המודל ניתן לשינוי דרך OPENAI_MODEL. ברירת המחדל מאזנת בין איכות, מהירות ועלות.
# ===============================

DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
DEFAULT_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "900"))


# ===============================
# שלב: בדיקת זמינות API
# הסבר: מאפשר לאפליקציה לדעת אם ניתן להפעיל תשובה מנוסחת עם LLM.
# ===============================

def is_openai_configured() -> bool:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    return bool(api_key and api_key != "your_api_key_here")


# ===============================
# שלב: יצירת לקוח OpenAI
# הסבר: יוצר לקוח רשמי של OpenAI שקורא את המפתח מתוך משתני הסביבה.
# ===============================

def create_openai_client() -> OpenAI:
    if not is_openai_configured():
        raise RuntimeError("OPENAI_API_KEY לא מוגדר. יש להוסיף מפתח לקובץ .env או למשתני הסביבה.")

    return OpenAI()


# ===============================
# שלב: הכנת קטעי מקור עבור המודל
# הסבר: ממיר את תוצאות השליפה לפורמט קצר וברור, כולל מזהה מקור ועמוד.
# ===============================

def format_sources_for_prompt(sources: list[dict[str, Any]], max_sources: int = 5) -> str:
    formatted_sources = []

    for index, source in enumerate(sources[:max_sources], start=1):
        formatted_sources.append(
            "\n".join(
                [
                    f"[מקור {index}]",
                    f"מסמך: {source.get('source', 'לא ידוע')}",
                    f"עמוד: {source.get('page', 'לא ידוע')}",
                    f"טקסט: {source.get('excerpt', '')}",
                ]
            )
        )

    return "\n\n".join(formatted_sources)


# ===============================
# שלב: הנחיות מערכת ל-LLM
# הסבר: ההנחיות מגבילות את המודל לענות רק מתוך המקורות, בעברית, ועם ציטוטי עמוד.
# ===============================

def build_llm_instructions() -> str:
    return """
אתה עוזר מחקרי למרכז ידע ארגוני.
עליך לענות בעברית בלבד.
עליך להשתמש רק במידע שמופיע במקורות שסופקו לך.
אסור לך להוסיף ידע כללי, הנחות, פרשנויות או נתונים שאינם מופיעים במקורות.
כל טענה עובדתית חייבת לכלול ציטוט מקור בפורמט: (מקור: שם המסמך, עמ' מספר).
אם אין במקורות מספיק מידע כדי לענות, כתוב בבירור: "אין מספיק מידע במקורות שנשלפו כדי לענות על השאלה."
אם יש כמה מקורות, שלב ביניהם בזהירות וציין מקור לכל נקודה מרכזית.
מבנה תשובה מומלץ:
1. תשובה קצרה וישירה.
2. נקודות מרכזיות.
3. מקורות.
"""


# ===============================
# שלב: בניית בקשת המשתמש למודל
# הסבר: מחבר את השאלה ואת המקורות שנשלפו לבקשה אחת שמועברת ל-Responses API.
# ===============================

def build_llm_input(query: str, sources: list[dict[str, Any]]) -> str:
    return f"""
שאלה:
{query}

מקורות שנשלפו:
{format_sources_for_prompt(sources)}

נסח תשובה מקצועית, קצרה וברורה לקורא עברי.
"""


# ===============================
# שלב: בדיקות בסיסיות לתשובת LLM
# הסבר: בודק אם התשובה כוללת עברית, ציטוטי עמוד, ואם קיימים מקורות בפועל.
# ===============================

def validate_llm_answer(answer: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
    source_pages = {str(source.get("page")) for source in sources if source.get("page") is not None}
    cited_pages = set(re.findall(r"עמ['׳]?\s*(\d+)", answer))

    return {
        "has_hebrew": bool(re.search(r"[\u0590-\u05FF]", answer)),
        "has_sources": bool(sources),
        "has_page_citations": bool(cited_pages),
        "cited_known_pages": bool(source_pages.intersection(cited_pages)) if cited_pages else False,
        "source_pages": sorted(source_pages),
        "cited_pages": sorted(cited_pages),
        "source_only_prompt_applied": True,
    }


# ===============================
# שלב: ניסוח תשובה עם LLM
# הסבר: מקבל תוצאת Agent קיימת, שולח למודל רק את המקורות שנשלפו, ומחזיר תשובה מנוסחת.
# ===============================

def synthesize_answer_with_llm(agent_result: dict[str, Any], model: str | None = None) -> dict[str, Any]:
    sources = agent_result.get("sources", [])

    if not sources:
        answer = "אין מספיק מידע במקורות שנשלפו כדי לענות על השאלה."
        return {
            **agent_result,
            "llm_answer": answer,
            "llm_model": model or DEFAULT_OPENAI_MODEL,
            "llm_validation": validate_llm_answer(answer, sources),
        }

    client = create_openai_client()
    selected_model = model or DEFAULT_OPENAI_MODEL

    response = client.responses.create(
        model=selected_model,
        instructions=build_llm_instructions(),
        input=build_llm_input(agent_result["query"], sources),
        max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
    )

    answer = response.output_text.strip()

    return {
        **agent_result,
        "llm_answer": answer,
        "llm_model": selected_model,
        "llm_validation": validate_llm_answer(answer, sources),
    }
