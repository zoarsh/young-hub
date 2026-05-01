# סט שאלות הערכה

תיקייה זו משמשת לבדיקת אמינות מערכת ה-RAG.

הקבצים המרכזיים:

```text
questions.jsonl
expected_answers.jsonl
policy_questions.jsonl
policy_expected_answers.jsonl
practitioner_questions.jsonl
practitioner_expected_answers.jsonl
population_gap_questions.jsonl
population_gap_expected_answers.jsonl
policy_practice_questions.jsonl
policy_practice_expected_answers.jsonl
evaluation_results.jsonl
```

`questions.jsonl` כולל שאלות בלבד, עם מזהה שאלה, קטגוריה וסוג שאלה צפוי.

`expected_answers.jsonl` כולל תשובות צפויות בלבד, עם אותו מזהה שאלה ועם עובדות מפתח לבדיקה.

`policy_questions.jsonl` כולל שאלות שמעניינות קובעי מדיניות במשרדי הממשלה השונים.

`policy_expected_answers.jsonl` כולל תשובות צפויות לשאלות קובעי המדיניות.

`practitioner_questions.jsonl` כולל שאלות שמעניינות אנשי מקצוע בשטח שעובדים עם צעירים.

`practitioner_expected_answers.jsonl` כולל תשובות צפויות לשאלות אנשי המקצוע.

`population_gap_questions.jsonl` כולל שאלות על פערים והבדלים בין קבוצות אוכלוסייה.

`population_gap_expected_answers.jsonl` כולל תשובות צפויות לשאלות הפערים בין קבוצות אוכלוסייה.

`policy_practice_questions.jsonl` כולל שאלות מרכזיות למדיניות ולפרקטיקה.

`policy_practice_expected_answers.jsonl` כולל תשובות צפויות לשאלות המדיניות והפרקטיקה.

`evaluation_results.jsonl` ייווצר כאשר מריצים את סקריפט ההערכה.

ההפרדה בין שאלות לתשובות מאפשרת להריץ את המערכת על השאלות בלי לחשוף לה את התשובות הצפויות, ואז להשוות בין תוצאת המערכת לבין התשובה הידנית.

דוגמת הרצה לסט הכללי:

```powershell
python scripts\evaluate_retrieval.py
```

דוגמת הרצה לסט קובעי מדיניות:

```powershell
python scripts\evaluate_retrieval.py --questions policy_questions.jsonl --answers policy_expected_answers.jsonl --output policy_evaluation_results.jsonl
```

דוגמת הרצה לסט אנשי מקצוע בשטח:

```powershell
python scripts\evaluate_retrieval.py --questions practitioner_questions.jsonl --answers practitioner_expected_answers.jsonl --output practitioner_evaluation_results.jsonl
```

דוגמת הרצה לסט פערים בין קבוצות אוכלוסייה:

```powershell
python scripts\evaluate_retrieval.py --questions population_gap_questions.jsonl --answers population_gap_expected_answers.jsonl --output population_gap_evaluation_results.jsonl
```

דוגמת הרצה לסט מדיניות ופרקטיקה:

```powershell
python scripts\evaluate_retrieval.py --questions policy_practice_questions.jsonl --answers policy_practice_expected_answers.jsonl --output policy_practice_evaluation_results.jsonl
```
