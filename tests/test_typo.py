import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.sql_system import SQLSystem

def test_typo():
    print("Testing Typo Query...")
    sql = SQLSystem()
    
    # User query
    query = "how many students have been blaced?"
    
    print(f"\nQuery: {query}")
    # Force use of _generate_sql by inspecting it directly if public, or just running the callable
    # _generate_sql is private, but we can check if regex catches it.
    
    entities = sql.extract_entities(query)
    print(f"Entities (Regex): {entities}")
    
    # If entities is empty/insufficient, it calls _generate_sql internally.
    # I'll temporarily patch the _generate_sql to print the valid SQL query 
    # OR I can just look at the result.
    # The result '0' means the query likely ran successfully but found nothing.
    
    result = sql(query)
    print(f"Result: {result}")
    
    sql.close()

if __name__ == "__main__":
    test_typo()
