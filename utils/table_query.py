# utils/table_query.py

def query_tables_with_question(tables, question):
    results = []
    for table in tables:
        if not table.empty and any(isinstance(c, str) for c in table.columns):
            combined_columns = " ".join(str(col).lower() for col in table.columns)
            if any(k in combined_columns for k in ["net", "profit", "2023", "2024", "revenue"]):
                results.append(table.to_string(index=False))
    if not results:
        return "No relevant financial data found for your question."
    return "\n\n".join(results[:2])  # top 2 tables
