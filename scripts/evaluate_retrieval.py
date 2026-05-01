# ===============================
# כותרת: הערכת אמינות ראשונית למערכת RAG
# הסבר: סקריפט זה מריץ את שאלות ההערכה מול סוכן הידע, שומר את תשובות המערכת,
#       ומחשב מדדים בסיסיים להשוואה מול תשובות צפויות.
# ===============================

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path
from typing import Any


# ===============================
# שלב: חיבור לתיקיית הפרויקט
# הסבר: מאפשר לייבא את מודולי ה-RAG המקומיים גם כאשר הסקריפט רץ מתיקיית scripts.
# ===============================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
EVALUATION_DIR = DATA_DIR / "evaluation"

if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from rag_agent_v0 import answer_with_knowledge_agent
from rag_index import load_vector_db


# ===============================
# שלב: הגדרת ברירות מחדל
# הסבר: כברירת מחדל ההערכה רצה מול המאגר המשולב וכל סט השאלות הכללי.
# ===============================

DEFAULT_COLLECTION_ID = "reports_d035e9af6fe6"
DEFAULT_QUESTIONS_FILE = "questions.jsonl"
DEFAULT_ANSWERS_FILE = "expected_answers.jsonl"
DEFAULT_OUTPUT_FILE = "evaluation_results.jsonl"


# ===============================
# שלב: קריאת פרמטרים
# הסבר: מאפשר להריץ הערכה על סטים שונים, למשל סט כללי או סט לקובעי מדיניות.
# ===============================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="הרצת הערכת אמינות למערכת RAG")
    parser.add_argument("--collection-id", default=DEFAULT_COLLECTION_ID, help="מזהה מאגר Chroma להרצה")
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS_FILE, help="שם קובץ השאלות בתוך data/evaluation")
    parser.add_argument("--answers", default=DEFAULT_ANSWERS_FILE, help="שם קובץ התשובות הצפויות בתוך data/evaluation")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FILE, help="שם קובץ התוצאות בתוך data/evaluation")
    return parser.parse_args()


# ===============================
# שלב: קריאת JSONL
# הסבר: טוען קבצי שאלות ותשובות שבהם כל שורה היא רשומת JSON עצמאית.
# ===============================

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


# ===============================
# שלב: כתיבת JSONL
# הסבר: שומר את תוצאות ההערכה באופן שורתית כדי שיהיה קל לבדוק ולעבד בהמשך.
# ===============================

def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


# ===============================
# שלב: בדיקת עובדות מפתח
# הסבר: בודק כמה מערכי המפתח הידניים מופיעים בתשובת המערכת או במקטעי המקור שנשלפו.
# ===============================

def count_key_fact_matches(key_facts: list[str], generated_text: str) -> tuple[int, list[str]]:
    matched = [fact for fact in key_facts if fact in generated_text]
    return len(matched), matched


# ===============================
# שלב: הערכת שאלה אחת
# הסבר: מריץ את ה-Agent על שאלה אחת ומחזיר תוצאה עם מקורות ומדדי התאמה ראשוניים.
# ===============================

def evaluate_question(vector_db: Any, question_record: dict[str, Any], expected_record: dict[str, Any]) -> dict[str, Any]:
    result = answer_with_knowledge_agent(vector_db, question_record["question"])
    generated_text = result["answer"] + " " + " ".join(source["excerpt"] for source in result["sources"])
    match_count, matched_facts = count_key_fact_matches(expected_record.get("key_facts", []), generated_text)

    return {
        "id": question_record["id"],
        "category": question_record["category"],
        "question": question_record["question"],
        "expected_query_type": question_record["query_type"],
        "detected_query_type": result["query_type"],
        "query_type_match": question_record["query_type"] == result["query_type"],
        "key_facts_total": len(expected_record.get("key_facts", [])),
        "key_facts_matched": match_count,
        "matched_facts": matched_facts,
        "sources": result["sources"],
        "generated_answer": result["answer"],
        "expected_answer": expected_record["expected_answer"],
    }


# ===============================
# שלב: הרצת הערכה מלאה
# הסבר: מריץ את כל השאלות מול המאגר ושומר קובץ תוצאות לבדיקה ידנית.
# ===============================

def main() -> None:
    args = parse_args()
    questions = read_jsonl(EVALUATION_DIR / args.questions)
    expected_answers = {record["id"]: record for record in read_jsonl(EVALUATION_DIR / args.answers)}
    vector_db = load_vector_db(args.collection_id)

    results = [
        evaluate_question(vector_db, question, expected_answers[question["id"]])
        for question in questions
    ]

    output_path = EVALUATION_DIR / args.output
    write_jsonl(output_path, results)

    query_type_accuracy = sum(1 for result in results if result["query_type_match"]) / len(results)
    fact_matches = sum(result["key_facts_matched"] for result in results)
    fact_total = sum(result["key_facts_total"] for result in results)
    fact_coverage = fact_matches / fact_total if fact_total else 0

    print("הערכת האמינות הסתיימה.")
    print(f"שאלות שנבדקו: {len(results)}")
    print(f"דיוק זיהוי סוג שאלה: {query_type_accuracy:.2%}")
    print(f"כיסוי עובדות מפתח: {fact_coverage:.2%}")
    print(f"קובץ תוצאות: {output_path}")


# ===============================
# שלב: נקודת כניסה
# הסבר: מריץ את ההערכה רק כאשר מפעילים את הקובץ ישירות.
# ===============================

if __name__ == "__main__":
    main()
