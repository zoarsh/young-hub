# ===============================
# כותרת: בניית קבצי תמיכה ומאגר וקטורי
# הסבר: סקריפט זה מכין מראש את כל הקבצים הדרושים לאפליקציה: טקסט לפי עמוד,
#       chunks, metadata, manifest ו-Chroma DB לשימוש מהיר באפליקציית Streamlit.
# ===============================

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ===============================
# שלב: חיבור לתיקיית הפרויקט
# הסבר: מאפשר להריץ את הסקריפט מכל מקום ועדיין לייבא את מודול ההכנה מתוך data.
# ===============================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from rag_index import build_collection, list_pdf_reports


# ===============================
# שלב: קריאת פרמטרים מהמשתמש
# הסבר: מאפשר לבנות אינדקס לדוח ברירת המחדל, לכל הדוחות, או לרשימת דוחות ספציפית.
# ===============================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="בניית מאגר ידע RAG מתוך קובצי PDF")
    parser.add_argument(
        "--all",
        action="store_true",
        help="בונה אוסף הכולל את כל קובצי ה-PDF בתיקיית data/raw_reports",
    )
    parser.add_argument(
        "--reports",
        nargs="*",
        default=["Young_Statistical_2025.pdf"],
        help="רשימת שמות קובצי PDF לבנייה",
    )
    return parser.parse_args()


# ===============================
# שלב: בחירת הדוחות לבנייה
# הסבר: מחליט אילו דוחות ייכנסו לאוסף לפי הפרמטרים שנשלחו לסקריפט.
# ===============================

def resolve_report_names(args: argparse.Namespace) -> list[str]:
    if args.all:
        return [path.name for path in list_pdf_reports()]

    return args.reports


# ===============================
# שלב: הרצת הבנייה
# הסבר: מפעיל את תהליך ההכנה המלא ומדפיס סיכום קצר בסיום.
# ===============================

def main() -> None:
    report_names = resolve_report_names(parse_args())

    if not report_names:
        raise ValueError("לא נבחרו דוחות לבנייה.")

    print("מתחיל בניית מאגר ידע...")
    print(f"דוחות: {', '.join(report_names)}")

    summary = build_collection(report_names)

    print("הבנייה הסתיימה.")
    print(f"מזהה אוסף: {summary['collection_id']}")
    print(f"עמודים שחולצו: {summary['pages_count']}")
    print(f"קטעים שנוצרו: {summary['chunks_count']}")
    print(f"קבצי עיבוד: {summary['processed_dir']}")
    print(f"Vector DB: {summary['vector_db_dir']}")


# ===============================
# שלב: נקודת כניסה
# הסבר: מריץ את הסקריפט רק כאשר מפעילים אותו ישירות מהטרמינל.
# ===============================

if __name__ == "__main__":
    main()
