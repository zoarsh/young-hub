# ===============================
# כותרת: אפליקציית Streamlit לסוכן ידע מחקרי
# הסבר: קובץ זה בונה ממשק אינטראקטיבי בעברית לשאלות על דוחות מחקר,
#       תוך שימוש בקבצי עיבוד ובמאגר וקטורי שמורים מראש.
# ===============================

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


# ===============================
# שלב: הגדרת נתיבי הפרויקט
# הסבר: מחבר את האפליקציה למודולים המקומיים שמנהלים את ה-Agent ואת מאגר הידע.
# ===============================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from rag_agent_v0 import answer_with_knowledge_agent
from rag_index import build_collection_id, is_collection_ready, list_pdf_reports, load_or_build_vector_db
from rag_llm_answer import DEFAULT_OPENAI_MODEL, is_openai_configured, synthesize_answer_with_llm


# ===============================
# שלב: הגדרת עמוד ועיצוב
# הסבר: מגדיר תצוגה רחבה, תמיכה בעברית וכרטיסי מקורות קריאים.
# ===============================

st.set_page_config(
    page_title="מרכז ידע ארגוני חכם",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        direction: rtl;
        text-align: right;
        font-family: "Segoe UI", Arial, sans-serif;
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 750;
        margin-bottom: 0.35rem;
    }

    .subtitle {
        color: #475569;
        font-size: 1.05rem;
        margin-bottom: 1.25rem;
    }

    .status-box {
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        background: #f8fafc;
        margin-bottom: 1rem;
    }

    .answer-box {
        border: 1px solid #d9e2ec;
        border-radius: 8px;
        padding: 1.1rem 1.25rem;
        background: #ffffff;
        line-height: 1.75;
    }

    .source-card {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.9rem 1rem;
        background: #f8fafc;
        margin-bottom: 0.75rem;
    }

    .small-label {
        color: #64748b;
        font-size: 0.88rem;
        margin-bottom: 0.25rem;
    }

    div[data-testid="stChatInput"] textarea {
        direction: rtl;
        text-align: right;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ===============================
# שלב: טעינת מאגר הידע עם cache
# הסבר: Streamlit שומר את המאגר בזיכרון כדי שלא נטען את Chroma מחדש בכל אינטראקציה.
# ===============================

@st.cache_resource(show_spinner=False)
def cached_load_or_build_vector_db(report_names: tuple[str, ...]):
    return load_or_build_vector_db(report_names)


# ===============================
# שלב: תרגום סוג שאלה לעברית
# הסבר: מציג למשתמש תווית פשוטה במקום שם פנימי באנגלית.
# ===============================

def query_type_label(query_type: str) -> str:
    labels = {
        "definition": "הגדרה",
        "data": "נתונים",
        "comparison": "השוואה",
        "summary": "סיכום",
    }
    return labels.get(query_type, query_type)


# ===============================
# שלב: הצגת סטטוס המאגר
# הסבר: מראה האם האינדקס כבר מוכן או שייבנה בהרצה הנוכחית.
# ===============================

def render_collection_status(report_names: tuple[str, ...]) -> None:
    collection_id = build_collection_id(report_names)
    status = "מוכן ונטען מדיסק" if is_collection_ready(collection_id) else "ייבנה בהרצה הנוכחית"

    st.markdown(
        f"""
        <div class="status-box">
            <strong>מזהה מאגר:</strong> {collection_id}<br>
            <strong>סטטוס:</strong> {status}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===============================
# שלב: הצגת מקורות
# הסבר: מציג קטעי מקור עם שם הדוח, מספר העמוד וציון הרלוונטיות.
# ===============================

def render_sources(sources: list[dict[str, object]]) -> None:
    if not sources:
        st.info("לא נמצאו מקורות להצגה.")
        return

    for source in sources:
        st.markdown(
            f"""
            <div class="source-card">
                <div class="small-label">מקור: {source["source"]} | עמ' {source["page"]} | ציון: {source["score"]}</div>
                <div>{source["excerpt"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ===============================
# שלב: כותרת ראשית
# הסבר: מציג את מטרת המערכת לפני אזור השאלות.
# ===============================

st.markdown('<div class="main-title">מרכז ידע ארגוני חכם</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">שאלות בעברית על דוחות מחקר, עם תשובות שמבוססות רק על מסמכי המקור.</div>',
    unsafe_allow_html=True,
)


# ===============================
# שלב: בחירת דוחות
# הסבר: מאפשר למשתמש לבחור דוח אחד או כמה דוחות לבניית מרחב השליפה.
# ===============================

available_reports = list_pdf_reports()

with st.sidebar:
    st.header("הגדרות מאגר הידע")

    if not available_reports:
        st.error("לא נמצאו קובצי PDF בתיקייה data/raw_reports.")
        st.stop()

    default_report = "Young_Statistical_2025.pdf"
    default_selection = [default_report] if default_report in [path.name for path in available_reports] else [available_reports[0].name]

    selected_reports = st.multiselect(
        "בחרי דוחות לשאילה",
        options=[path.name for path in available_reports],
        default=default_selection,
    )

    st.caption("מומלץ להתחיל מדוח אחד. בחירה בכמה דוחות תיצור מאגר נפרד עבור השילוב.")

    # ===============================
    # שלב: בחירת מצב תשובה
    # הסבר: מאפשר לעבור בין תשובה בסיסית מה-Agent לבין ניסוח מתקדם בעזרת LLM.
    # ===============================

    answer_mode = st.radio(
        "מצב תשובה",
        options=["תשובה בסיסית ללא LLM", "תשובה מנוסחת עם LLM"],
        index=0,
    )

    if answer_mode == "תשובה מנוסחת עם LLM":
        if is_openai_configured():
            st.success(f"LLM פעיל: {DEFAULT_OPENAI_MODEL}")
        else:
            st.warning("כדי להפעיל LLM יש להחליף את מפתח הדמה בקובץ .env במפתח OpenAI אמיתי.")

if not selected_reports:
    st.warning("בחרי לפחות דוח אחד כדי לבנות או לטעון את מאגר הידע.")
    st.stop()

selected_reports_tuple = tuple(selected_reports)
render_collection_status(selected_reports_tuple)


# ===============================
# שלב: טעינת המאגר הווקטורי
# הסבר: טוען מאגר קיים או בונה אותו פעם אחת מתוך קובצי ה-PDF וקבצי העיבוד.
# ===============================

with st.spinner("טוען או בונה את מאגר הידע הסמנטי..."):
    vector_db = cached_load_or_build_vector_db(selected_reports_tuple)


# ===============================
# שלב: שמירת היסטוריית שיחה
# הסבר: שומר את השאלות והתשובות במהלך השימוש הנוכחי באפליקציה.
# ===============================

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ===============================
# שלב: קבלת שאלה והצגת תשובה
# הסבר: מעביר את השאלה לסוכן הידע ומציג תשובה, סוג שאלה ומקורות.
# ===============================

question = st.chat_input("כתבי שאלה בעברית על הדוח...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("מחפש במקורות ומדרג תשובה..."):
            result = answer_with_knowledge_agent(vector_db, question)

        displayed_answer = result["answer"]

        # ===============================
        # שלב: ניסוח תשובה בעזרת LLM
        # הסבר: אם המשתמש בחר מצב LLM, המודל מקבל רק את המקורות שנשלפו ומחזיר תשובה עברית עם ציטוטי עמוד.
        # ===============================

        if answer_mode == "תשובה מנוסחת עם LLM":
            if not is_openai_configured():
                st.error("מצב LLM נבחר, אבל OPENAI_API_KEY עדיין לא מוגדר בקובץ .env.")
            else:
                with st.spinner("מנסח תשובה בעברית בעזרת LLM, רק על בסיס המקורות שנשלפו..."):
                    result = synthesize_answer_with_llm(result)
                    displayed_answer = result["llm_answer"]

        st.markdown(f"**סוג השאלה שזוהה:** {query_type_label(result['query_type'])}")
        st.markdown(
            f"""
            <div class="answer-box">
                {displayed_answer.replace(chr(10), "<br>")}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if "llm_validation" in result:
            validation = result["llm_validation"]
            with st.expander("בדיקות איכות לתשובת LLM"):
                st.write(f"תשובה בעברית: {'כן' if validation['has_hebrew'] else 'לא'}")
                st.write(f"קיימים מקורות שנשלפו: {'כן' if validation['has_sources'] else 'לא'}")
                st.write(f"קיימים ציטוטי עמוד: {'כן' if validation['has_page_citations'] else 'לא'}")
                st.write(f"העמודים שצוטטו תואמים למקורות שנשלפו: {'כן' if validation['cited_known_pages'] else 'לא'}")
                st.write("הנחיית שימוש במקורות בלבד: הופעלה")

        st.subheader("מקורות")
        render_sources(result["sources"])

    st.session_state.messages.append({"role": "assistant", "content": displayed_answer})
