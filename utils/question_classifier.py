# utils/question_classifier.py

def is_numeric_question(question: str) -> bool:
    keywords = [
        "net income", "revenue", "profit", "loss", "compare",
        "increase", "decrease", "2023", "2024", "amount",
        "percentage", "growth", "drop", "financial", "report"
    ]
    return any(k in question.lower() for k in keywords)
