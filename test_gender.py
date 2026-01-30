import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.sql_system import SQLSystem

def test_gender():
    sql = SQLSystem()
    print("Testing Gender Queries...")
    
    queries = [
        "how many girls placed?",
        "list male students in cse",
        "count of female students"
    ]
    
    for q in queries:
        print(f"\nQuery: {q}")
        print(f"Entities: {sql.extract_entities(q)}")
        print(f"Result: {sql(q)[:100]}...") # Print first 100 chars

    sql.close()

if __name__ == "__main__":
    test_gender()
