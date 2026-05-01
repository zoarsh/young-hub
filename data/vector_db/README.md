# תיקיית Vector DB

תיקייה זו מכילה מאגרי Chroma שנבנו מתוך קבצי ה-chunks.

כל תת-תיקייה מייצגת אוסף דוחות אחר, לפי מזהה מסוג:

```text
reports_<hash>
```

האפליקציה `app.py` טוענת את המאגר המתאים מתוך תיקייה זו כדי לבצע שליפה סמנטית מהירה.

מבנה פנימי טיפוסי של Chroma כולל:

```text
chroma.sqlite3
<uuid>/
```

אין לערוך ידנית את קובצי Chroma. אם רוצים לבנות מחדש את המאגר, יש להריץ:

```powershell
python scripts\build_knowledge_base.py --reports Young_Statistical_2025.pdf
```

או לכל הדוחות:

```powershell
python scripts\build_knowledge_base.py --all
```
