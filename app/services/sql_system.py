"""
SQL System for Student Queries
Handles natural language queries about student data
"""
import sqlite3
import pandas as pd
import re
from pathlib import Path

import os
import requests
import json

class SQLSystem:
    def __init__(self, db_path=None, ollama_model=None, ollama_url=None):
        """Initialize SQL system with student database"""
        project_root = Path(__file__).resolve().parent.parent.parent
        self.db_path = db_path or str(project_root / 'app/database/students.db')
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        
        # Ollama Configuration
        self.ollama_model = ollama_model or os.environ.get('OLLAMA_MODEL', 'llama3.2:3b')
        self.ollama_url = ollama_url or os.environ.get('OLLAMA_URL', 'http://127.0.0.1:11434/api/generate')
        
        # Get table schema
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
        
        # Extract roll number (student ID)
        roll_match = re.search(r'\b(\d{2}[A-Z]\d{2}[A-Z]\d{4})\b', query) or re.search(r'\b(\d{5,10})\b', query)
        if roll_match:
            entities['roll_no'] = roll_match.group(1)
        
        # Extract branch (department) - comprehensive list with aliases
        branch_aliases = {
            'cse': 'CSE', 'computer science': 'CSE', 'cs': 'CSE',
            'cse-aiml': 'CSE-AIML', 'aiml': 'CSE-AIML', 'ai': 'CSE-AIML', 'ml': 'CSE-AIML',
            'artificial intelligence': 'CSE-AIML', 'machine learning': 'CSE-AIML',
            'cse-ds': 'CSE-DS', 'csd': 'CSE-DS', 'data science': 'CSE-DS',
            'csm': 'CSM',
            'ece': 'ECE', 'electronics': 'ECE', 'electronics and communication': 'ECE',
            'eee': 'EEE', 'electrical': 'EEE', 'electronics and electrical': 'EEE',
            'me': 'ME', 'mech': 'ME', 'mechanical': 'ME', 'mechanical engineering': 'ME',
            'ce': 'CE', 'civil': 'CE', 'civil engineering': 'CE',
            'it': 'IT', 'information technology': 'IT',
            'mba': 'MBA', 'management': 'MBA',
        }
        # Sort by length descending so longer aliases match first (e.g., 'data science' before 'data')
        for alias in sorted(branch_aliases.keys(), key=len, reverse=True):
            if re.search(r'\b' + re.escape(alias) + r'\b', query_lower):
                entities['branch'] = branch_aliases[alias]
                break
        
        # Extract name (capitalized words)
        name_match = re.search(r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b', query)
        if name_match:
            entities['name'] = name_match.group(1)
        
        # Extract CGPA condition
        cgpa_match = re.search(r'cgpa\s*([><=]+)\s*(\d+\.?\d*)', query_lower)
        if cgpa_match:
            entities['cgpa_operator'] = cgpa_match.group(1)
            entities['cgpa_value'] = float(cgpa_match.group(2))
        
        # Extract placement status - check negatives FIRST
        if 'not placed' in query_lower or 'unplaced' in query_lower or "didn't" in query_lower or "didnt" in query_lower:
            entities['placed'] = False
        elif 'placed' in query_lower:
            entities['placed'] = True
            
        # Extract Gender
        if re.search(r'\b(girl|girls|female|females|woman|women|ladies)\b', query_lower):
            entities['gender'] = 'Female'
        elif re.search(r'\b(boy|boys|male|males|man|men|guys)\b', query_lower):
            entities['gender'] = 'Male'
            
        # Extract Average GPA intent
        if 'average' in query_lower and ('gpa' in query_lower or 'cgpa' in query_lower):
            entities['average_gpa'] = True
        
        return entities
    
    def build_sql_query(self, entities):
        """Build SQL query from extracted entities"""
        if entities.get('average_gpa'):
            base_query = "SELECT AVG(CGPA) as 'Average CGPA' FROM students"
        else:
            base_query = "SELECT * FROM students"
            
        conditions = []
        
        if 'roll_no' in entities:
            conditions.append(f"\"ROLL NO\" = '{entities['roll_no']}'")
        
        if 'branch' in entities:
            conditions.append(f"UPPER(BRANCH) = '{entities['branch']}'")
        
        if 'name' in entities:
            conditions.append(f"NAME LIKE '%{entities['name']}%'")
        
        if 'cgpa_operator' in entities:
            op = entities['cgpa_operator']
            val = entities['cgpa_value']
            conditions.append(f"CGPA {op} {val}")
        
        if 'placed' in entities:
            if entities['placed']:
                conditions.append("\"COMPANY PLACED\" IS NOT NULL AND \"COMPANY PLACED\" != 'Not Placed'")
            else:
                conditions.append("(\"COMPANY PLACED\" IS NULL OR \"COMPANY PLACED\" = 'Not Placed')")
        
        if 'gender' in entities:
             conditions.append(f"GENDER = '{entities['gender']}'")
        
        if conditions:
            base_query += " WHERE " + " AND ".join(conditions)
        
        # Increased limit for better aggregation
        base_query += " LIMIT 1000"
        return base_query
    
    def _validate_sql(self, sql_query):
        """Safety filter: validate SQL query to prevent destructive operations.
        
        Returns:
            tuple: (is_safe: bool, reason: str)
        """
        if not sql_query or not sql_query.strip():
            return False, "Empty SQL query"
        
        sql_upper = sql_query.strip().upper()
        
        # Block destructive operations
        blocked_keywords = ['DROP', 'DELETE', 'ALTER', 'UPDATE', 'INSERT', 'CREATE', 'TRUNCATE', 'REPLACE', 'ATTACH', 'DETACH']
        for keyword in blocked_keywords:
            if re.search(r'\b' + keyword + r'\b', sql_upper):
                return False, f"Blocked: '{keyword}' operations are not allowed"
        
        # Only allow SELECT statements
        if not sql_upper.lstrip().startswith('SELECT'):
            return False, "Only SELECT queries are allowed"
        
        # Ensure query targets the students table
        if 'STUDENTS' not in sql_upper:
            return False, "Query must target the 'students' table"
        
        # Enforce LIMIT cap to prevent full-table dumps
        if 'LIMIT' not in sql_upper:
            sql_query = sql_query.rstrip().rstrip(';') + ' LIMIT 1000'
        
        return True, sql_query

    def _get_fallback_links(self, query: str) -> str:
        """Return topic-relevant links even when SQL generation fails."""
        q = query.lower()
        if any(k in q for k in ['placed', 'placement', 'package', 'company', 'recruit']):
            return "\n\n⚠ I couldn't query the specific data right now. You may find what you need here:\n- [Placement Cell](https://tkrcet.ac.in/placement/)"
        if any(k in q for k in ['fee', 'payment', 'cost']):
            return "\n\n⚠ I couldn't query fees right now. Please see:\n- [Fee Details](https://tkrcet.ac.in/fees-structure/)"
        if any(k in q for k in ['result', 'exam', 'marks', 'score']):
            return "\n\n⚠ I couldn't query that right now. Check results at:\n- [Exam Branch](https://tkrcet.ac.in/exam-branch/)"
        return "\n\n⚠ I couldn't query the database right now. Visit [TKRCET](https://tkrcet.ac.in/) for more info."

    def _generate_sql(self, query):
        """Generate SQL query using LLM"""
        print(f"  [SQL] Generating SQL for: {query}")
        
        schema_desc = "\n".join([f'- "{col}"' for col in self.columns])
        
        # Add hint for Gender & Columns
        hints = "GENDER values are 'Male' and 'Female'. 'COMPANY PLACED' contains company name or 'Not Placed'. To find placed students, check \"COMPANY PLACED\" != 'Not Placed'."
        
        prompt = f"""You are a SQL expert. Convert the user's natural language query into a valid SQLite query for the 'students' table.

Table Schema (Exact Column Names):
{schema_desc}

Hints: {hints}

Examples:
1. User: "students with cgpa > 9"
   SQL: SELECT * FROM students WHERE "CGPA" > 9 LIMIT 50
2. User: "how many students passed in 2024?"
   SQL: SELECT COUNT(*) FROM students WHERE "PASSED YEAR" = 2024
3. User: "how many students are placed?"
   SQL: SELECT COUNT(*) FROM students WHERE "COMPANY PLACED" IS NOT NULL AND "COMPANY PLACED" != 'Not Placed'

Query: {query}

Rules:
1. Return ONLY the raw SQL query. No markdown, no explanations.
2. ALWAYS quote all column names using double quotes (e.g., "PASSED YEAR" not PASSED_YEAR).
3. Use LIMIT 50 for listings.
4. No LIMIT for aggregations.

SQL:"""
        
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0} # Deterministic
                },
                timeout=45  # Increased from 30 to 45s for slow local machines
            )
            if response.status_code == 200:
                sql = response.json().get('response', '').strip()
                # cleanup markdown if present
                sql = re.sub(r'```sql', '', sql)
                sql = re.sub(r'```', '', sql).strip()
                return sql
        except Exception as e:
            print(f"⚠ LLM SQL Gen Error: {e}")
            return None
        return None
    
    def query_students(self, query, chat_history=None):
        """
        Execute natural language query on student database
        
        Args:
            query: Natural language query
            chat_history: Optional list of previous conversation turns for context
        
        Examples:
        - "What is the CGPA of student 12345?"
        - "List all students in CSE department"
        - "Show students with CGPA > 8.5"
        - "Find student named John Doe"
        """
        query_lower = query.lower()
        
        # Context-aware follow-up question handling
        if chat_history and len(chat_history) > 0:
            # Check if this is a follow-up question (e.g., "which companies?", "how many?")
            follow_up_indicators = ['which', 'what about', 'how many of them', 'from them', 'those']
            is_follow_up = any(indicator in query_lower for indicator in follow_up_indicators)
            
            if is_follow_up and len(query.split()) < 6:  # Short questions are likely follow-ups
                # Look for previous placement/student query in history
                for turn in reversed(chat_history[-4:]):  # Check last 4 turns
                    if turn.get('role') == 'user':
                        prev_query = turn.get('message', '').lower()
                        # If previous query was about placements, append context
                        if any(kw in prev_query for kw in ['placed', 'placement', 'company', 'students']):
                            query = f"{prev_query} {query}"  # Merge context
                            query_lower = query.lower()
                            print(f"  [SQL Context] Merged with previous query: {query}")
                            break
        
        # Special case: top companies query
        if 'companies' in query_lower or 'recruiter' in query_lower:
            try:
                df = pd.read_sql_query(
                    """SELECT "COMPANY PLACED", COUNT(*) as count 
                       FROM students 
                       WHERE "COMPANY PLACED" IS NOT NULL AND "COMPANY PLACED" != 'Not Placed'
                       GROUP BY "COMPANY PLACED"
                       ORDER BY count DESC
                       LIMIT 10""", 
                    self.conn
                )
                if len(df) == 0:
                    return "No placement data available."
                
                response = "Top Recruiting Companies:\n\n"
                for _, row in df.iterrows():
                    response += f"• {row['COMPANY PLACED']}: {row['count']} students\n"
                return response
            except Exception as e:
                return f"Error fetching company data: {e}"

        # Special case: placements by year — "how many students placed in 2024?"
        year_match = re.search(r'\b(20\d{2})\b', query_lower)
        if year_match and any(kw in query_lower for kw in ['placed', 'placement', 'got job', 'recruited', 'how many']):
            year = int(year_match.group(1))
            try:
                df_total = pd.read_sql_query(
                    f"SELECT COUNT(*) as total FROM students WHERE `PASSED YEAR` = {year}",
                    self.conn
                )
                df_placed = pd.read_sql_query(
                    f"""SELECT COUNT(*) as placed FROM students
                       WHERE `PASSED YEAR` = {year}
                       AND `COMPANY PLACED` IS NOT NULL
                       AND `COMPANY PLACED` != 'Not Placed'
                       AND `COMPANY PLACED` != ''""",
                    self.conn
                )
                total = df_total.iloc[0]['total']
                placed = df_placed.iloc[0]['placed']
                if total == 0:
                    return f"No student data found for the year {year}."
                rate = round((placed / total) * 100, 1) if total > 0 else 0
                return (
                    f"**Placement Statistics for {year}:**\n"
                    f"\n• Total Students: {total}"
                    f"\n• Students Placed: {placed}"
                    f"\n• Placement Rate: {rate}%"
                    f"\n\n- [Placement Cell](https://tkrcet.ac.in/placement/)"
                )
            except Exception as e:
                return f"Error fetching year-wise placement data: {e}"

        # Special case: highest / max package query
        if any(kw in query_lower for kw in ['highest package', 'max package', 'maximum package', 'highest salary', 'top package', 'highest ctc']):
            try:
                df = pd.read_sql_query(
                    """SELECT NAME, BRANCH, "COMPANY PLACED", "PACKAGE"
                       FROM students
                       WHERE "PACKAGE" IS NOT NULL AND "PACKAGE" != '' AND "PACKAGE" != '0'
                       ORDER BY CAST(REPLACE(REPLACE("PACKAGE", 'LPA', ''), ' ', '') AS REAL) DESC
                       LIMIT 1""",
                    self.conn
                )
                if len(df) == 0:
                    return "Package data is not available in the database."
                row = df.iloc[0]
                return (
                    f"🏆 **Highest Package:**\n"
                    f"• **Name:** {row.get('NAME', 'N/A')}\n"
                    f"• **Branch:** {row.get('BRANCH', 'N/A')}\n"
                    f"• **Company:** {row.get('COMPANY PLACED', 'N/A')}\n"
                    f"• **Package:** {row.get('PACKAGE', 'N/A')}\n"
                    f"\n_(Individual data shared with permission. Contact admin for verification.)_"
                )
            except Exception as e:
                # Fallback: just return count of placed students with any package info
                return f"Package details are not available in structured form. Contact the Placement Cell for more information.\n\n- [Placement Cell](https://tkrcet.ac.in/placement/)"
        
        # Special case: how many X students placed in SPECIFIC company
        # e.g. "how many cse students placed in jbm group?"
        company_match = re.search(
            r'(?:placed in|got placed in|students? in|placed at)\s+(.+?)(?:\?|$|in \d{4})',
            query_lower
        )
        branch_in_query = None
        for alias, code in [
            ('cse', 'CSE'), ('ece', 'ECE'), ('eee', 'EEE'), ('it', 'IT'),
            ('mba', 'MBA'), ('civil', 'CE'), ('mech', 'ME'), ('data science', 'CSE-DS'),
            ('aiml', 'CSE-AIML'), ('csd', 'CSE-DS'), ('csm', 'CSM')
        ]:
            if alias in query_lower:
                branch_in_query = code
                break
        
        if company_match and branch_in_query:
            company_name = company_match.group(1).strip().rstrip('?').strip()
            try:
                df = pd.read_sql_query(
                    f"""SELECT COUNT(*) as count FROM students
                       WHERE UPPER(BRANCH) = '{branch_in_query}'
                       AND LOWER("COMPANY PLACED") LIKE '%{company_name.lower()}%'""",
                    self.conn
                )
                count = df.iloc[0]['count']
                if count == 0:
                    return f"No {branch_in_query} students found placed in a company matching '{company_name}'."
                return (
                    f"• **{count} {branch_in_query} student{'s' if count != 1 else ''}** placed in **{company_name.title()}**\n"
                    f"\n- [Placement Cell](https://tkrcet.ac.in/placement/)"
                )
            except Exception as e:
                return f"Error querying placement data: {e}"

        # Extract entities from query
        entities = self.extract_entities(query)
        
        if not entities:
            # Fallback to Text-to-SQL if regex finds nothing
            print("  [SQL] No entities found, trying Text-to-SQL...")
            sql_query = self._generate_sql(query)
            if not sql_query:
                return "I couldn't understand the student query. Please specify student ID, name, department, or conditions." + self._get_fallback_links(query)
            # Post-validate LLM-generated SQL
            is_safe, result = self._validate_sql(sql_query)
            if not is_safe:
                print(f"  [SQL Safety] LLM query blocked: {result}")
                return "I couldn't safely process that query. Please try rephrasing with specific student details."
            sql_query = result  # Use potentially modified query (with LIMIT added)
        else:
             # Build SQL query from regex entities
             sql_query = self.build_sql_query(entities)
             # Validate regex-built query too (defense in depth)
             is_safe, result = self._validate_sql(sql_query)
             if not is_safe:
                 print(f"  [SQL Safety] Regex query blocked: {result}")
                 return "Error processing query. Please try again."
             sql_query = result
        
        try:
            # Execute query
            result_df = pd.read_sql_query(sql_query, self.conn)
            
            if len(result_df) == 0:
                return "No students found matching your criteria."
            
            # Handle Aggregation Results (COUNT, AVG, etc.)
            if len(result_df) == 1 and len(result_df.columns) == 1:
                col_name = result_df.columns[0]
                value = result_df.iloc[0, 0]
                
                # Format floats (like Average GPA) to 2 decimal places
                if isinstance(value, float):
                    value = round(value, 2)
                    
                return f"Result: {value}"
            
            # PRIVACY: Only show aggregate data for general queries
            # Individual records only for specific student lookup
            if len(result_df) == 1 and 'roll_no' in entities:
                # Specific student lookup - show limited details
                student = result_df.iloc[0]
                response = f"Student Information:\n"
                safe_cols = ['NAME', 'BRANCH', 'CGPA', 'COMPANY PLACED']
                for col in safe_cols:
                    if col in result_df.columns:
                        response += f"  {col}: {student[col]}\n"
                return response.strip()
            else:
                # General query - show ONLY aggregate summary (PRIVACY)
                total = len(result_df)
                
                # Calculate statistics
                response = f"Summary ({total} students found):\n\n"
                
                # Placement stats
                if 'COMPANY PLACED' in result_df.columns:
                    placed = result_df[result_df['COMPANY PLACED'].notna() & (result_df['COMPANY PLACED'] != 'Not Placed')]
                    response += f"• Placed students: {len(placed)} out of {total}\n"
                    
                    # Top companies
                    if len(placed) > 0:
                        companies = placed['COMPANY PLACED'].value_counts().head(3)
                        response += f"• Top recruiters: {', '.join(companies.index.tolist())}\n"
                
                # CGPA stats
                if 'CGPA' in result_df.columns:
                    avg_cgpa = result_df['CGPA'].mean()
                    max_cgpa = result_df['CGPA'].max()
                    response += f"• Average CGPA: {avg_cgpa:.2f}\n"
                    response += f"• Highest CGPA: {max_cgpa:.2f}\n"
                
                # Branch distribution
                if 'BRANCH' in result_df.columns:
                    branches = result_df['BRANCH'].value_counts().to_dict()
                    response += f"• Branch-wise: {', '.join([f'{k}: {v}' for k, v in branches.items()])}\n"
                
                response += "\n(Individual student data is protected. Please contact administration for specific records.)"
                
                # Append Related Links (Static Mapping)
                links = "\n\nRelated Links:"
                has_links = False
                
                if 'placed' in query.lower() or 'package' in query.lower() or 'company' in query.lower():
                    links += "\n- [Placement Cell](https://tkrcet.ac.in/placement/)"
                    has_links = True
                
                if 'exam' in query.lower() or 'result' in query.lower() or 'passed' in query.lower() or 'failed' in query.lower():
                    links += "\n- [Exam Results Portal](https://tkrcet.ac.in/exam-branch/results/)"
                    has_links = True
                    
                if 'fee' in query.lower() or 'payment' in query.lower():
                    links += "\n- [Fee Payment](https://tkrcet.ac.in/payment/)"
                    has_links = True
                
                if not has_links:
                    links += "\n- [TKRCET Home](https://tkrcet.ac.in/)"
                
                return response + links
        
        except Exception as e:
            return f"Error querying database: {str(e)}"
    
    def __call__(self, query, chat_history=None):
        """Make class callable"""
        return self.query_students(query, chat_history=chat_history)
    
    def close(self):
        """Close database connection"""
        self.conn.close()

if __name__ == "__main__":
    # Test the SQL system
    sql_system = SQLSystem()
    
    print("Testing SQL System...")
    print("="*70)
    
    # Test queries
    test_queries = [
        "What is the CGPA of student 12345?",
        "List all students in CSE department",
        "Show students with CGPA > 8.5",
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-"*70)
        result = sql_system(query)
        print(result)
        print()
    
    sql_system.close()
