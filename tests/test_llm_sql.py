import sys
from pathlib import Path

from app.services.sql_system import SQLSystem

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_llm_sql():
    print("Testing LLM SQL Generation...")
    sql = SQLSystem()

    # regex fails for this, should trigger LLM
    query = "average cgpa of male students"

    print(f"\nQuery: {query}")
    print(f"Entities (Regex): {sql.extract_entities(query)}")

    # extract_entities for this might catch 'gender': 'Male', and 'cgpa'
    # My regex for CGPA is for specific value conditions (cgpa > 8).
    # 'average cgpa' might NOT match entities['cgpa_condition'].
    # 'male' will match gender.

    # If ANY entity is matched, it uses regex path.
    # If 'average' is not in regex logic,
    # regex path will just filter by gender 'Male'
    # and return ALL male students.

    # Then summary generator (in query_students)
    # calculates average.

    # So actually `average cgpa` works with regex path
    # because `query_students` computes summary stats.

    # Let's try something Regex definitely fails at.

    query2 = "how many students passed in 2023?"

    print(f"\nQuery 2: {query2}")

    # Should be empty
    print(f"Entities (Regex): {sql.extract_entities(query2)}")

    print(
        "Result:\n"
        f"{sql(query2)}"
    )

    # Try complex logic
    query3 = "students with cgpa greater than 9 in cse branch"

    # Regex DOES catch this.
    # We want to verify LLM fallback.

    print(f"\nQuery 3: {query3}")
    print(f"Entities (Regex): {sql.extract_entities(query3)}")
    print(f"Result:\n{sql(query3)}")

    sql.close()


if __name__ == "__main__":
    test_llm_sql()
