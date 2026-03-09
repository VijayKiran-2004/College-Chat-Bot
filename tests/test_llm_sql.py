import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.sql_system import SQLSystem  # noqa:E402


def test_llm_sql():
    print("Testing LLM SQL Generation...")
    sql = SQLSystem()

    # regex fails for this, should trigger LLM
    query = "average cgpa of male students"

    print(f"\nQuery: {query}")
    print(f"Entities (Regex): {sql.extract_entities(query)}")
    # extract_entities for this might catch 'gender': 'Male', and 'cgpa'... wait.
    # My regex for CGPA is for specific value conditions (cgpa > 8).
    # 'average cgpa' might NOT match entities['cgpa_condition'].
    # 'male' will match gender.
    # If ANY entity is matched, it uses regex path. I need to be careful.
    # If 'average' is not in regex logic,
    # the regex path will just filter by gender 'Male'
    # and return ALL male students.
    # Then the summary generator (in query_students) calculates average.
    # So actually `average cgpa` WORKS with regex path because
    # `query_students` computes summary stats!

    # Let's try something that Regex definitely fails at.
    # "List students who passed in 2023"
    # I don't have 'passed year' in regex.

    query2 = "how many students passed in 2023?"
    print(f"\nQuery 2: {query2}")
    print(f"Entities (Regex): {sql.extract_entities(query2)}")  # Should be empty
    print(f"Result:\n{sql(query2)}")  # Should use LLM and print SQL on error

    sql.close()


if __name__ == "__main__":
    test_llm_sql()
