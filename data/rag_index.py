# ===============================
# כותרת: הכנת מאגר ידע עבור RAG
# הסבר: קובץ זה מרכז את כל פעולות ההכנה: חילוץ טקסט מה-PDF, חלוקה לקטעים,
#       שמירת קבצי ביניים, בניית Chroma DB וטעינת המאגר עבור האפליקציה.
# ===============================

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings


# ===============================
# שלב: הגדרת נתיבים מרכזיים
# הסבר: מגדיר מיקום אחיד לדוחות המקור, לקבצי העיבוד ולמאגר הווקטורי.
# ===============================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = DATA_DIR / "raw_reports"
PROCESSED_DIR = DATA_DIR / "processed"
VECTOR_DB_DIR = DATA_DIR / "vector_db"

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 160
MIN_PAGE_CHARS = 80
MIN_CHUNK_CHARS = 120


# ===============================
# שלב: איתור דוחות PDF
# הסבר: מחזיר את רשימת הדוחות הזמינים בתיקיית המקור.
# ===============================

def list_pdf_reports() -> list[Path]:
    return sorted(REPORTS_DIR.glob("*.pdf"))


# ===============================
# שלב: יצירת מזהה אוסף
# הסבר: מזהה האוסף נבנה משמות הדוחות כדי לאפשר מאגר נפרד לכל שילוב דוחות.
# ===============================

def build_collection_id(report_names: list[str] | tuple[str, ...]) -> str:
    raw_names = "|".join(sorted(report_names))
    digest = hashlib.sha1(raw_names.encode("utf-8")).hexdigest()[:12]
    return f"reports_{digest}"


# ===============================
# שלב: ניקוי טקסט בסיסי
# הסבר: מאחד רווחים ושבירות שורה כדי לשפר את איכות ה-chunks ואת השליפה.
# ===============================

def clean_text(text: str) -> str:
    return " ".join(str(text or "").split())


# ===============================
# שלב: כתיבת JSONL
# הסבר: שומר רשומות טקסט בצורה קריאה ושורתית, נוחה לבדיקה ולשימוש חוזר.
# ===============================

def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


# ===============================
# שלב: קריאת JSONL
# הסבר: טוען קבצי ביניים שכבר נוצרו בלי לחלץ שוב את ה-PDF.
# ===============================

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


# ===============================
# שלב: כתיבת metadata לקובץ CSV
# הסבר: יוצר טבלה קצרה שמאפשרת לבחון אילו chunks נוצרו ומאיזה עמוד הם הגיעו.
# ===============================

def write_metadata_csv(path: Path, chunks: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["chunk_uid", "source", "page", "chunk_id", "text_length"]

    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for chunk in chunks:
            writer.writerow({field: chunk.get(field, "") for field in fieldnames})


# ===============================
# שלב: חילוץ עמודים מתוך PDF
# הסבר: מחלץ טקסט מכל עמוד ושומר מספר עמוד ושם מקור לציטוטים מדויקים.
# ===============================

def extract_pages_from_pdf(pdf_path: Path) -> list[dict[str, Any]]:
    reader = PdfReader(str(pdf_path))
    pages: list[dict[str, Any]] = []

    for index, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text())
        if len(text) < MIN_PAGE_CHARS:
            continue

        pages.append(
            {
                "source": pdf_path.name,
                "page": index,
                "text": text,
                "text_length": len(text),
            }
        )

    return pages


# ===============================
# שלב: חילוץ כל העמודים
# הסבר: עובר על כל הדוחות שנבחרו ומייצר מאגר עמודים אחיד.
# ===============================

def extract_pages(report_names: list[str] | tuple[str, ...]) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []

    for report_name in report_names:
        report_path = REPORTS_DIR / report_name
        if not report_path.exists():
            raise FileNotFoundError(f"לא נמצא קובץ PDF: {report_path}")
        pages.extend(extract_pages_from_pdf(report_path))

    return pages


# ===============================
# שלב: חלוקה לקטעי טקסט
# הסבר: יוצר chunks עם חפיפה, כדי שהשליפה תישאר ממוקדת אבל לא תאבד הקשר.
# ===============================

def split_pages_to_chunks(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict[str, Any]] = []

    for page in pages:
        page_chunks = splitter.split_text(page["text"])
        for chunk_index, chunk_text in enumerate(page_chunks):
            clean_chunk = clean_text(chunk_text)
            if len(clean_chunk) < MIN_CHUNK_CHARS:
                continue

            chunk_uid = f"{page['source']}::p{page['page']}::c{chunk_index}"
            chunks.append(
                {
                    "chunk_uid": chunk_uid,
                    "source": page["source"],
                    "page": page["page"],
                    "chunk_id": chunk_index,
                    "text": clean_chunk,
                    "text_length": len(clean_chunk),
                }
            )

    return chunks


# ===============================
# שלב: יצירת manifest
# הסבר: שומר תיאור קצר של האוסף שנבנה כדי שנדע באילו דוחות, מודל והגדרות השתמשנו.
# ===============================

def build_manifest(collection_id: str, report_names: list[str] | tuple[str, ...], pages_count: int, chunks_count: int) -> dict[str, Any]:
    return {
        "collection_id": collection_id,
        "reports": list(report_names),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "pages_count": pages_count,
        "chunks_count": chunks_count,
    }


# ===============================
# שלב: שמירת קבצי עיבוד
# הסבר: שומר pages.jsonl, chunks.jsonl, metadata.csv ו-manifest.json לבקרה ושימוש חוזר.
# ===============================

def save_processed_files(collection_id: str, report_names: list[str] | tuple[str, ...], pages: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> Path:
    output_dir = PROCESSED_DIR / collection_id
    output_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(output_dir / "pages.jsonl", pages)
    write_jsonl(output_dir / "chunks.jsonl", chunks)
    write_metadata_csv(output_dir / "metadata.csv", chunks)

    manifest = build_manifest(collection_id, report_names, len(pages), len(chunks))
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return output_dir


# ===============================
# שלב: טעינת מודל Embeddings
# הסבר: טוען מודל רב-לשוני שמתאים לעברית ומשמש גם לבנייה וגם לשליפה.
# ===============================

def load_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


# ===============================
# שלב: בניית Chroma DB
# הסבר: יוצר מאגר וקטורי שמור לדיסק מתוך ה-chunks וה-metadata.
# ===============================

def build_vector_db(collection_id: str, chunks: list[dict[str, Any]]) -> Chroma:
    if not chunks:
        raise ValueError("לא נמצאו chunks לבניית מאגר וקטורי.")

    texts = [chunk["text"] for chunk in chunks]
    metadatas = [
        {
            "chunk_uid": chunk["chunk_uid"],
            "source": chunk["source"],
            "page": chunk["page"],
            "chunk_id": chunk["chunk_id"],
            "text_length": chunk["text_length"],
        }
        for chunk in chunks
    ]

    persist_directory = VECTOR_DB_DIR / collection_id
    persist_directory.mkdir(parents=True, exist_ok=True)

    return Chroma.from_texts(
        texts=texts,
        embedding=load_embedding_model(),
        metadatas=metadatas,
        collection_name=collection_id,
        persist_directory=str(persist_directory),
    )


# ===============================
# שלב: טעינת Chroma DB קיים
# הסבר: טוען מאגר וקטורי שכבר נבנה מראש כדי שהאפליקציה תעלה מהר.
# ===============================

def load_vector_db(collection_id: str) -> Chroma:
    persist_directory = VECTOR_DB_DIR / collection_id
    if not persist_directory.exists():
        raise FileNotFoundError(f"לא נמצא מאגר וקטורי: {persist_directory}")

    return Chroma(
        collection_name=collection_id,
        embedding_function=load_embedding_model(),
        persist_directory=str(persist_directory),
    )


# ===============================
# שלב: בדיקה האם מאגר מוכן
# הסבר: בודק אם קיימים גם קבצי עיבוד וגם Chroma DB עבור אוסף הדוחות.
# ===============================

def is_collection_ready(collection_id: str) -> bool:
    processed_dir = PROCESSED_DIR / collection_id
    vector_dir = VECTOR_DB_DIR / collection_id

    return (
        (processed_dir / "pages.jsonl").exists()
        and (processed_dir / "chunks.jsonl").exists()
        and (processed_dir / "metadata.csv").exists()
        and (processed_dir / "manifest.json").exists()
        and vector_dir.exists()
    )


# ===============================
# שלב: בניית אוסף מלא
# הסבר: פעולה מקצה לקצה שמחלצת עמודים, יוצרת chunks, שומרת קבצים ובונה Chroma DB.
# ===============================

def build_collection(report_names: list[str] | tuple[str, ...]) -> dict[str, Any]:
    collection_id = build_collection_id(report_names)
    pages = extract_pages(report_names)
    chunks = split_pages_to_chunks(pages)
    processed_dir = save_processed_files(collection_id, report_names, pages, chunks)
    build_vector_db(collection_id, chunks)

    return {
        "collection_id": collection_id,
        "processed_dir": str(processed_dir),
        "vector_db_dir": str(VECTOR_DB_DIR / collection_id),
        "pages_count": len(pages),
        "chunks_count": len(chunks),
    }


# ===============================
# שלב: טעינה או בנייה לפי צורך
# הסבר: אם המאגר כבר מוכן הוא נטען מיד; אם לא, הוא נבנה פעם אחת ונשמר להמשך.
# ===============================

def load_or_build_vector_db(report_names: list[str] | tuple[str, ...]) -> Chroma:
    collection_id = build_collection_id(report_names)

    if not is_collection_ready(collection_id):
        build_collection(report_names)

    return load_vector_db(collection_id)
