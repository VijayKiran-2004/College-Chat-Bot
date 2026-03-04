"""
SQL System for Student Queries
Handles natural language queries about student data
"""

import os
import re
import sqlite3

import pandas as pd


class SQLSystem:

    def __init__(
        self,
        db_path="app/database/students.db",
        ollama_model=None,
        ollama_url=None,
    ):
        """Initialize SQL system with student database"""

        self.db_path = db_path
        self.conn = sqlite3.connect(
            db_path,
            check_same_thread=False,
        )

        self.ollama_model = ollama_model or os.environ.get(
            "OLLAMA_MODEL",
            "llama3.2:3b",
        )

        self.ollama_url = ollama_url or os.environ.get(
            "OLLAMA_URL",
            "http://127.0.0.1:11434/api/generate",
        )

        self.columns = self._get_columns()

    def _get_columns(self):
        """Get list of columns in students table"""

        cursor = self.conn.cursor()
        cursor.execute("PRAGMA table_info(students)")
        columns = [row[1] for row in cursor.fetchall()]
        return columns

    def extract_entities(self, query):
        """Extract entities from natural language query"""

        query_lower = query.lower()
        entities = {}

        roll_match = re.search(
            r"\b(\d{2}[A-Z]\d{2}[A-Z]\d{4})\b",
            query,
        ) or re.search(
            r"\b(\d{5,10})\b",
            query,
        )

        if roll_match:
            entities["roll_no"] = roll_match.group(1)

        branch_aliases = {
            "cse": "CSE",
            "computer science": "CSE",
            "cs": "CSE",
            "cse-aiml": "CSE-AIML",
            "aiml": "CSE-AIML",
            "ai": "CSE-AIML",
            "ml": "CSE-AIML",
            "artificial intelligence": "CSE-AIML",
            "machine learning": "CSE-AIML",
            "cse-ds": "CSE-DS",
            "csd": "CSE-DS",
            "data science": "CSE-DS",
            "csm": "CSM",
            "ece": "ECE",
            "electronics": "ECE",
            "electronics and communication": "ECE",
            "eee": "EEE",
            "electrical": "EEE",
            "electronics and electrical": "EEE",
            "me": "ME",
            "mech": "ME",
            "mechanical": "ME",
            "mechanical engineering": "ME",
            "ce": "CE",
            "civil": "CE",
            "civil engineering": "CE",
            "it": "IT",
            "information technology": "IT",
            "mba": "MBA",
            "management": "MBA",
        }

        for alias in sorted(
            branch_aliases.keys(),
            key=len,
            reverse=True,
        ):

            if re.search(
                r"\b" + re.escape(alias) + r"\b",
                query_lower,
            ):
                entities["branch"] = branch_aliases[alias]
                break

        name_match = re.search(
            r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b",
            query,
        )

        if name_match:
            entities["name"] = name_match.group(1)

        cgpa_match = re.search(
            r"cgpa\s*([><=]+)\s*(\d+\.?\d*)",
            query_lower,
        )

        if cgpa_match:
            entities["cgpa_operator"] = cgpa_match.group(1)
            entities["cgpa_value"] = float(cgpa_match.group(2))

        if (
            "not placed" in query_lower
            or "unplaced" in query_lower
            or "didn't" in query_lower
            or "didnt" in query_lower
        ):
            entities["placed"] = False

        elif "placed" in query_lower:
            entities["placed"] = True

        if re.search(
            r"\b(girl|girls|female|women)\b",
            query_lower,
        ):
            entities["gender"] = "Female"

        elif re.search(
            r"\b(boy|boys|male|men)\b",
            query_lower,
        ):
            entities["gender"] = "Male"

        if (
            "average" in query_lower
            and ("gpa" in query_lower or "cgpa" in query_lower)
        ):
            entities["average_gpa"] = True

        return entities

    def build_sql_query(self, entities):
        """Build SQL query from extracted entities"""

        if entities.get("average_gpa"):
            base_query = (
                "SELECT AVG(CGPA) as 'Average CGPA' "
                "FROM students"
            )
        else:
            base_query = "SELECT * FROM students"

        conditions = []

        if "roll_no" in entities:
            conditions.append(
                f"\"ROLL NO\" = '{entities['roll_no']}'"
            )

        if "branch" in entities:
            conditions.append(
                f"UPPER(BRANCH) = '{entities['branch']}'"
            )

        if "name" in entities:
            conditions.append(
                f"NAME LIKE '%{entities['name']}%'"
            )

        if "cgpa_operator" in entities:
            op = entities["cgpa_operator"]
            val = entities["cgpa_value"]
            conditions.append(f"CGPA {op} {val}")

        if "placed" in entities:

            if entities["placed"]:

                conditions.append(
                    "\"COMPANY PLACED\" IS NOT NULL "
                    "AND \"COMPANY PLACED\" != 'Not Placed'"
                )

            else:

                conditions.append(
                    "("
                    "\"COMPANY PLACED\" IS NULL "
                    "OR \"COMPANY PLACED\" = 'Not Placed'"
                    ")"
                )

        if "gender" in entities:
            conditions.append(
                f"GENDER = '{entities['gender']}'"
            )

        if conditions:
            base_query += " WHERE " + " AND ".join(conditions)

        base_query += " LIMIT 1000"

        return base_query

    def query_students(
        self,
        query,
        chat_history=None,
    ):
        """Execute natural language query on student database"""

        entities = self.extract_entities(query)

        sql_query = self.build_sql_query(entities)

        try:

            result_df = pd.read_sql_query(
                sql_query,
                self.conn,
            )

            if len(result_df) == 0:
                return "No students found."

            if (
                len(result_df) == 1
                and len(result_df.columns) == 1
            ):
                value = result_df.iloc[0, 0]

                if isinstance(value, float):
                    value = round(value, 2)

                return f"Result: {value}"

            return result_df.to_string(index=False)

        except Exception as e:

            return f"Database error: {str(e)}"

    def __call__(self, query, chat_history=None):
        return self.query_students(
            query,
            chat_history=chat_history,
        )

    def close(self):
        self.conn.close()


if __name__ == "__main__":

    sql_system = SQLSystem()

    print("Testing SQL System")
    print("=" * 70)

    test_queries = [
        "What is the CGPA of student 12345?",
        "List all students in CSE department",
        "Show students with CGPA > 8.5",
    ]

    for query in test_queries:

        print(f"\nQuery: {query}")
        print("-" * 70)

        result = sql_system(query)

        print(result)
        print()

    sql_system.close()
