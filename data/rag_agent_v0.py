# ===============================
# כותרת: סוכן ידע מבוסס RAG - גרסה ראשונה
# הסבר: קובץ זה מוסיף שכבת Agent מעל מנוע השליפה הקיים. הסוכן מזהה את סוג השאלה,
#       מתאים את אסטרטגיית השליפה, מדרג קטעים, ומחזיר תשובה עברית עם מקורות לפי עמוד.
# ===============================

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable


# ===============================
# שלב: הגדרת מבנה תוצאה אחיד
# הסבר: מבנה זה עוזר להציג לכל קורא מה נמצא, מאיזה מקור, ומה ציון הרלוונטיות שלו.
# ===============================

@dataclass
class AgentSource:
    text: str
    page: Any
    source: Any
    score: int


# ===============================
# שלב: זיהוי סוג השאלה
# הסבר: הסוכן מבחין בין שאלת הגדרה, נתונים, השוואה או סיכום כדי לבחור אסטרטגיית שליפה מתאימה.
# ===============================

def detect_query_type(query: str) -> str:
    query = normalize_spaces(query)

    definition_terms = ["מה ההגדרה", "הגדרה", "מה פירוש", "מה זה", "מוגדר"]
    data_terms = ["נתון", "נתונים", "שיעור", "אחוז", "אחוזים", "כמה", "מספר", "לוח", "תרשים", "%"]
    comparison_terms = ["השוואה", "להשוות", "פער", "פערים", "לעומת", "בהשוואה", "בין"]

    if contains_any(query, definition_terms):
        return "definition"
    if contains_any(query, comparison_terms):
        return "comparison"
    if contains_any(query, data_terms):
        return "data"

    return "summary"


# ===============================
# שלב: בחירת אסטרטגיית שליפה
# הסבר: לכל סוג שאלה מוגדרים מספר תוצאות לשליפה, מספר תוצאות ראשוניות, והאם להעדיף נתונים מספריים.
# ===============================

def build_retrieval_strategy(query_type: str) -> dict[str, Any]:
    strategies = {
        "definition": {
            "k": 4,
            "fetch_k": 20,
            "prefer_numeric": False,
            "prefer_textual": True,
            "title": "שאלת הגדרה",
        },
        "data": {
            "k": 5,
            "fetch_k": 40,
            "prefer_numeric": True,
            "prefer_textual": False,
            "title": "שאלת נתונים",
        },
        "comparison": {
            "k": 6,
            "fetch_k": 45,
            "prefer_numeric": True,
            "prefer_textual": False,
            "title": "שאלת השוואה",
        },
        "summary": {
            "k": 5,
            "fetch_k": 30,
            "prefer_numeric": False,
            "prefer_textual": True,
            "title": "שאלת סיכום",
        },
    }

    return strategies.get(query_type, strategies["summary"])


# ===============================
# שלב: ניקוי בסיסי של טקסטים
# הסבר: הניקוי משפר קריאות ומונע מצב שבו קטעים עם רווחים מיותרים מקבלים משקל לא נכון.
# ===============================

def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


# ===============================
# שלב: בדיקת הופעת מילים
# הסבר: פונקציה קטנה שמרכזת בדיקת מילות מפתח כדי לשמור על קוד קריא ופשוט.
# ===============================

def contains_any(text: str, terms: Iterable[str]) -> bool:
    return any(term in text for term in terms)


# ===============================
# שלב: חילוץ מונחי חיפוש מהשאלה
# הסבר: המונחים המרכזיים מהשאלה משמשים לדירוג נוסף מעבר לשליפה הסמנטית.
# ===============================

def extract_query_terms(query: str) -> list[str]:
    stop_words = {
        "מה", "מי", "על", "של", "עם", "את", "זה", "זו", "האם", "יש", "אין",
        "כמה", "לפי", "בין", "אילו", "איזה", "איזו", "בישראל", "בקרב",
    }
    terms = re.findall(r"[\u0590-\u05FFA-Za-z0-9%\-]+", query)
    return [term for term in terms if len(term) > 2 and term not in stop_words]


# ===============================
# שלב: זיהוי קטעים חלשים
# הסבר: קטע חלש הוא קטע קצר מדי, טכני מדי, או כזה שנראה כמו תוכן עניינים ולא כמו מידע מחקרי.
# ===============================

def is_weak_chunk(text: str) -> bool:
    clean_text = normalize_spaces(text)

    if len(clean_text) < 180:
        return True

    weak_terms = ["תוכן עניינים", "רשימת לוחות", "רשימת תרשימים", "מקורות", "נספח"]
    if contains_any(clean_text, weak_terms):
        return True

    return False


# ===============================
# שלב: זיהוי טבלאות
# הסבר: שאלות סיכום והגדרה לרוב צריכות טקסט מילולי, בעוד שאלות נתונים והשוואה יכולות ליהנות מטבלאות.
# ===============================

def is_probably_table(text: str) -> bool:
    clean_text = normalize_spaces(text)
    numbers = re.findall(r"\d+(?:\.\d+)?%?", clean_text)
    table_terms = ["לוח", "סך", "גברים", "נשים", "סה\"כ", "אחוזים"]

    return len(numbers) >= 12 and contains_any(clean_text, table_terms)


# ===============================
# שלב: זיהוי קטעים עם נתונים
# הסבר: קטע עם שנים, אחוזים או מספרים מקבל עדיפות כאשר המשתמש מבקש נתונים או השוואה.
# ===============================

def has_data_signals(text: str) -> bool:
    return bool(re.search(r"\d{4}|\d+(?:\.\d+)?\s?%|\d+(?:\.\d+)?", text))


# ===============================
# שלב: דירוג קטעים
# הסבר: הדירוג משלב בין התאמה לשאלה, סימני נתונים, התאמה לנושא, ואיכות בסיסית של הטקסט.
# ===============================

def score_document(doc: Any, query: str, query_type: str, strategy: dict[str, Any]) -> int:
    text = normalize_spaces(getattr(doc, "page_content", ""))
    query_terms = extract_query_terms(query)
    score = 0

    for term in query_terms:
        if term in text:
            score += 4

    if strategy["prefer_numeric"] and has_data_signals(text):
        score += 8

    if strategy["prefer_textual"] and not is_probably_table(text):
        score += 5

    if query_type in {"data", "comparison"} and is_probably_table(text):
        score += 4

    if contains_any(text, ["ישראל", "OECD", "צעירים", "צעירות"]):
        score += 3

    if is_weak_chunk(text):
        score -= 10

    return score


# ===============================
# שלב: שליפה סמנטית מהמאגר
# הסבר: הסוכן משתמש ב-Vector DB הקיים מהמחברת ומבצע שליפה סמנטית, לא חיפוש מילות מפתח בלבד.
# ===============================

def retrieve_candidate_documents(vector_db: Any, query: str, strategy: dict[str, Any]) -> list[Any]:
    return vector_db.similarity_search(query, k=strategy["fetch_k"])


# ===============================
# שלב: סינון ודירוג תוצאות
# הסבר: אחרי השליפה הסמנטית, הסוכן מסנן קטעים חלשים ומחזיר את הקטעים החזקים ביותר לפי סוג השאלה.
# ===============================

def select_best_sources(
    documents: list[Any],
    query: str,
    query_type: str,
    strategy: dict[str, Any],
) -> list[AgentSource]:
    selected: list[AgentSource] = []

    for doc in documents:
        text = normalize_spaces(getattr(doc, "page_content", ""))

        if is_weak_chunk(text):
            continue

        if strategy["prefer_textual"] and is_probably_table(text):
            continue

        metadata = getattr(doc, "metadata", {}) or {}
        selected.append(
            AgentSource(
                text=text,
                page=metadata.get("page", "לא ידוע"),
                source=metadata.get("source", "לא ידוע"),
                score=score_document(doc, query, query_type, strategy),
            )
        )

    selected.sort(key=lambda item: item.score, reverse=True)
    return selected[: strategy["k"]]


# ===============================
# שלב: יצירת תמצית מקורית מתוך קטע
# הסבר: מאחר שגרסה זו אינה משתמשת במודל שפה, היא מציגה משפטים קצרים מתוך המקורות עצמם בלבד.
# ===============================

def source_excerpt(text: str, max_chars: int = 420) -> str:
    clean_text = normalize_spaces(text)

    if len(clean_text) <= max_chars:
        return clean_text

    return clean_text[:max_chars].rsplit(" ", 1)[0] + "..."


# ===============================
# שלב: בניית תשובה מובנית בעברית
# הסבר: התשובה כוללת אבחון סוג השאלה, נקודות מרכזיות מתוך המקורות, וציטוטי עמוד ברורים.
# ===============================

def build_structured_answer(query: str, query_type: str, sources: list[AgentSource]) -> dict[str, Any]:
    if not sources:
        return {
            "query": query,
            "query_type": query_type,
            "answer": "לא נמצאו קטעים מספיק חזקים במקורות הקיימים כדי לענות על השאלה.",
            "sources": [],
        }

    opening_by_type = {
        "definition": "להלן הגדרה או הסבר מתוך המקורות שנשלפו:",
        "data": "להלן הנתונים המרכזיים שנמצאו במקורות:",
        "comparison": "להלן נקודות ההשוואה המרכזיות שנמצאו במקורות:",
        "summary": "להלן סיכום מבוסס מקורות:",
    }

    bullets = []
    for index, source in enumerate(sources[:3], start=1):
        bullets.append(
            f"{index}. {source_excerpt(source.text)} "
            f"(מקור: {source.source}, עמ' {source.page})"
        )

    answer = opening_by_type.get(query_type, opening_by_type["summary"])
    answer += "\n" + "\n".join(bullets)

    return {
        "query": query,
        "query_type": query_type,
        "answer": answer,
        "sources": [
            {
                "page": source.page,
                "source": source.source,
                "score": source.score,
                "excerpt": source_excerpt(source.text, max_chars=260),
            }
            for source in sources
        ],
    }


# ===============================
# שלב: פונקציית Agent מקצה לקצה
# הסבר: זו הפונקציה המרכזית להרצה במחברת. היא מקבלת שאלה בעברית ומחזירה תשובה עם מקורות.
# ===============================

def answer_with_knowledge_agent(vector_db: Any, query: str) -> dict[str, Any]:
    query_type = detect_query_type(query)
    strategy = build_retrieval_strategy(query_type)
    candidates = retrieve_candidate_documents(vector_db, query, strategy)
    sources = select_best_sources(candidates, query, query_type, strategy)

    return build_structured_answer(query, query_type, sources)


# ===============================
# שלב: הדפסה נוחה למחברת
# הסבר: פונקציה זו מציגה את תשובת הסוכן בצורה קריאה עבור בעלי עניין שאינם טכניים.
# ===============================

def print_agent_answer(agent_result: dict[str, Any]) -> None:
    query_type_labels = {
        "definition": "הגדרה",
        "data": "נתונים",
        "comparison": "השוואה",
        "summary": "סיכום",
    }

    print("================================")
    print("תשובת סוכן הידע")
    print("================================")
    print(f"שאלה: {agent_result['query']}")
    print(f"סוג שאלה: {query_type_labels.get(agent_result['query_type'], agent_result['query_type'])}")
    print()
    print(agent_result["answer"])
    print()
    print("מקורות לפי עמוד:")

    for source in agent_result["sources"]:
        print(f"- {source['source']}, עמ' {source['page']} | ציון רלוונטיות: {source['score']}")


# ===============================
# שלב: דוגמת שימוש במחברת
# הסבר: לאחר שהמחברת בנתה את vector_db, ניתן להריץ את שלוש השורות הבאות בתא חדש.
# ===============================

EXAMPLE_NOTEBOOK_USAGE = """
from rag_agent_v0 import answer_with_knowledge_agent, print_agent_answer

result = answer_with_knowledge_agent(
    vector_db,
    "מה ידוע על צעירים שאינם עובדים ואינם לומדים בישראל?"
)

print_agent_answer(result)
"""
