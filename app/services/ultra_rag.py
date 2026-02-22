"""
UltraRAG System - Modern MCP-based RAG implementation
Replaces the old custom RAG system with UltraRAG framework
"""

import json
import requests
import sys
from pathlib import Path

# Fix Windows encoding
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


import os

# Fix for WinError 1114 (DLL Initialization Failed)
# Must be set before importing libraries that use Torch/ChromaDB
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

try:
    import torch
except ImportError:
    pass # Torch might not be installed, but if it is, we need to load it early

# ============================================================
# KNOWLEDGE BASE - Instant answers for critical facts
# ============================================================
KNOWLEDGE_BASE = {
    "personnel": {
        "principal": "Dr. D. V. Ravi Shankar",
        "vice_principal": "Dr. A. Suresh Rao (also HoD of CSE & Dean Academics)",
        "dean_academics": "Dr. A. Suresh Rao (also Vice Principal & HoD of CSE)",
        "secretary": "Dr. T. Harinath Reddy",
        "chairman": "Sri. Teegala Krishna Reddy",
        "hod": {
            "cse": "Dr. A. Suresh Rao",
            "cse-aiml": "Dr. B. Sunil Srinivas",
            "csm": "Dr. B. Sunil Srinivas",
            "cse-ds": "Dr. V. Krishna",
            "csd": "Dr. V. Krishna",
            "ece": "Dr. D. Nageshwar Rao",
            "eee": "Dr. K. Raju",
            "it": "Dr. R. Muruanantham",
            "mech": "Mr. D. Rushi Kumar",
            "civil": "Mr. K.V.R Satya Sai",
            "mba": "Dr. K. Gyaneswari"
        }
    },
    "timings": {
        "working_hours": "9:40 AM to 4:20 PM (Monday-Saturday)",
        "lunch_break": "12:40 PM to 1:20 PM"
    },
    "history": {
        "established": "2002",
        "affiliation": "JNTUH (Jawaharlal Nehru Technological University Hyderabad)",
        "status": "Autonomous (UGC confirmed)",
        "location": "Meerpet, Hyderabad - 500097, Telangana",
        "campus_size": "20 acres"
    },
    "society": {
        "name": "TKR Educational Society",
        "full_form": "Teegala Krishna Reddy",
        "colleges": [
            "TKR College of Engineering and Technology (TKRCET) - Autonomous",
            "Teegala Krishna Reddy Engineering College (TKREC)",
            "TKR College of Pharmacy (TKRCOP)",
            "TKR Institute of Management and Science (TKRIMS)"
        ],
        "chairman": "Sri. Teegala Krishna Reddy"
    },
    "admissions": {
        "process": "Admissions are through TS EAPCET counseling for B.Tech, PGCET for M.Tech, and direct admission for MBA. Visit the admissions office or website for detailed procedure.",
        "eligibility": "10+2 with Physics, Chemistry, and Mathematics for B.Tech. Graduation in relevant field for M.Tech/MBA.",
        "contact": "Visit https://tkrcet.ac.in/admissions for admission details and fee structure."
    },
    "courses": {
        "ug": ["CSE", "CSE-AIML", "CSE-DS", "ECE", "EEE", "IT", "Mechanical", "Civil"],
        "pg": ["M.Tech in CSE", "M.Tech in Power Electronics", "MBA"],
        "total": "8 UG programs and 3 PG programs"
    },
    "facilities": {
        "main": "State-of-the-art labs, smart classrooms, Wi-Fi campus, digital library, hostel (boys & girls), transport, sports ground, auditorium, NCC, incubation center, and medical facilities.",
        "special": "Virtual labs, R&D center (21,000 sq ft), industry partnerships with ECIL and others.",
        "transport": {
            "details": "TKRCET provides a fleet of buses connecting all major parts of Hyderabad city to the campus. Transport is safe, reliable, and available for both students and staff.",
            "routes": "Buses operate from Dilshuknagar, LB Nagar, Secunderabad, Uppal, Mehdipatnam, Kukatpally, and other key areas.",
            "contact": "For specific route map and fee details, please visit the Transport Section in the Administrative Block."
        },
        "canteen": {
            "name": "College Canteen",
            "description": "The college canteen provides hygienic food, snacks, and beverages at subsidized rates. It is a clean and spacious area where students can relax and refuel.",
            "menu": "South Indian, North Indian, Chinese snacks, and beverages.",
            "timings": "Open during college hours (including lunch break)."
        }
    },
    "activities": {
        "ncc": {
            "name": "National Cadet Corps (NCC)",
            "description": "TKRCET has a vibrant NCC unit (Army Wing) that instills discipline, patriotism, and leadership qualities in students. Cadets participate in regular drills, camps, and social service activities.",
            "benefits": "NCC 'B' and 'C' certificates provide advantages in higher education admissions and government job selections (especially Defence)."
        },
        "nss": {
            "name": "National Service Scheme (NSS)",
            "description": "The NSS unit at TKRCET encourages students to serve the community through activities like blood donation camps, village adoption, tree plantation, and health awareness drives.",
            "motto": "Not Me But You"
        },
        "campus_life": {
            "overview": "TKRCET offers a vibrant campus life with a perfect blend of academics and extra-curricular activities.",
            "events": "Annual cultural fest 'SISIR', technical fest 'MEDHA', sports meets, and various club activities.",
            "clubs": "Student clubs for coding, robotics, literature, photography, and music.",
            "environment": "Ragging-free, eco-friendly, and safe campus environment."
        }
    },
    "accreditation": {
        "naac": "A+ Grade",
        "nba": "NBA Accredited",
        "approvals": "AICTE approved, UGC recognized 2(f) & 12(B)"
    },
    "fees": {
        "btech": "Tuition fee for B.Tech is approximately ₹85,000 - ₹1,00,000 per year (depending on the quota and branch).",
        "mtech": "Tuition fee for M.Tech is approximately ₹57,000 per year.",
        "mba": "Tuition fee for MBA is approximately ₹54,000 per year.",
        "hostel": "Hostel fee is around ₹70,000 - ₹80,000 per year (including mess).",
        "transport": "Transport fee varies by distance, ranging from ₹18,000 to ₹35,000 per year.",
        "note": "Fees are subject to change as per government regulations. Contact accounts department for exact figures."
    },
    "exam_info": {
        "exam_schedule": "Exam schedules are announced by the Examination Branch. Please check the college notice board, website, or contact your department for specific dates.",
        "exam_timings": "Exam timings vary by semester and course. Please refer to the exam timetable published by the college on the notice board or website.",
        "recent_events": "For information about recent or upcoming events, please check the college website, notice boards, or contact the Student Activities Office.",
        "upcoming_events": "For information about upcoming events, please check the college website, notice boards, or contact the Student Activities Office."
    }
}


class UltraRAGSystem:
    """
    UltraRAG-based RAG system for college-buddy chatbot
    Uses MCP architecture with FAISS + BM25 hybrid retrieval
    """
    
    def __init__(
        self,
        corpus_path=None,
        ollama_model=None,
        ollama_url=None,
    ):
        # Configuration from arguments or environment variables
        self.ollama_model = ollama_model or os.environ.get('OLLAMA_MODEL', 'llama3.2:3b')
        self.ollama_url = ollama_url or os.environ.get('OLLAMA_URL', 'http://localhost:11434/api/generate')
        
        # Calculate robust absolute paths
        # Project root is 3 levels up from this file (app/services/ultra_rag.py -> app/services -> app -> root)
        self.project_root = Path(__file__).resolve().parent.parent.parent
        
        # Default corpus path relative to project root
        if corpus_path is None:
            self.corpus_path = str(self.project_root / 'app/database/vectordb/corpus_ultrarag.jsonl')
        else:
            self.corpus_path = corpus_path
        
        # Load corpus for retrieval
        self.documents = self._load_corpus()
        print(f"✓ Loaded {len(self.documents)} documents")
        
        # Initialize retrieval components
        self.collection = None # Default to None
        try:
            import chromadb
            from chromadb.utils import embedding_functions
            
            print("Connecting to ChromaDB...")
            chroma_db_path = self.project_root / 'app/database/vectordb/chroma'
            self.chroma_client = chromadb.PersistentClient(path=str(chroma_db_path))
            
            # Use same embedding model as ingestion
            ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
            
            self.collection = self.chroma_client.get_collection(
                name="college_data",
                embedding_function=ef
            )
            print("✓ Connected to ChromaDB")
            
        except Exception as e:
            print(f"⚠ Error initializing retrieval: {e}")
            print("Run 'python scripts/ingest.py' to populate the database.")
            self.collection = None # Ensure it is None on failure
        
        # Test Ollama connection
        self.ollama_available = self._test_ollama()
        
        # Response cache for common queries
        self.response_cache = {}
        
        # Initialize KB semantic matching
        print("Building KB semantic index...")
        from sentence_transformers import SentenceTransformer
        self.kb_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self._build_kb_index()
        print("✓ KB semantic index ready")
        
        # Load SQL statistics into Knowledge Base
        self._load_sql_stats()
        
        print("✓ UltraRAGSystem ready!\n")
    
    def _load_sql_stats(self):
        """Load aggregate statistics from SQL database into Knowledge Base"""
        try:
            from app.services.sql_system import SQLSystem
            import pandas as pd
            sql = SQLSystem()
            print("Loading SQL statistics...")
            
            # Total students
            df_total = pd.read_sql_query("SELECT COUNT(*) as count FROM students", sql.conn)
            total_students = df_total.iloc[0]['count']
            
            # Placed students
            df_placed = pd.read_sql_query(
                "SELECT COUNT(*) as count FROM students WHERE \"COMPANY PLACED\" IS NOT NULL AND \"COMPANY PLACED\" != 'Not Placed'", 
                sql.conn
            )
            placed_count = df_placed.iloc[0]['count']
            
            # Top companies
            df_companies = pd.read_sql_query(
                """SELECT "COMPANY PLACED", COUNT(*) as count 
                   FROM students 
                   WHERE "COMPANY PLACED" IS NOT NULL AND "COMPANY PLACED" != 'Not Placed'
                   GROUP BY "COMPANY PLACED"
                   ORDER BY count DESC
                   LIMIT 3""",
                sql.conn
            )
            top_companies = ", ".join([f"{row['COMPANY PLACED']} ({row['count']})" for _, row in df_companies.iterrows()])
            
            # Inject into KNOWLEDGE_BASE
            KNOWLEDGE_BASE['statistics'] = {
                "total_students": str(total_students),
                "placed_students": str(placed_count),
                "top_recruiters": top_companies,
                "placement_rate": f"{int((placed_count/total_students)*100)}%" if total_students > 0 else "N/A"
            }
            
            sql.close()
            print(f"✓ SQL Stats loaded: {total_students} students, {placed_count} placed")
            
        except Exception as e:
            print(f"⚠ Could not load SQL stats: {e}")
            # Fallback defaults
            KNOWLEDGE_BASE['statistics'] = {
                "total_students": "1600+",
                "placed_students": "Many",
                "top_recruiters": "TCS, Wipro, Infosys",
                "placement_rate": "High"
            }
    
    def _random_response(self, value, key_type, extra_info=None):
        """Generate a random friendly response wrapper"""
        import random
        
        templates = {
            "principal": [
                f"The Principal of TKRCET is **{value}**.",
                f"Dr. **{value}** is our respected Principal.",
                f"That would be **{value}**!",
                f"Currently, **{value}** serves as the Principal."
            ],
            "vice_principal": [
                f"The Vice Principal is **{value}**.",
                f"**{value}** is the Vice Principal of our college."
            ],
            "secretary": [
                f"The Secretary is **{value}**.",
                f"**{value}** holds the position of Secretary."
            ],
            "chairman": [
                f"The Chairman is **{value}**.",
                f"Our Chairman is **{value}**."
            ],
            "dean": [
                f"The Dean of Academics is **{value}**.",
                f"**{value}** is the Dean of Academics."
            ],
            "hod": [
                f"The HOD of **{extra_info}** is **{value}**.",
                f"**{value}** heads the **{extra_info}** department.",
                f"For **{extra_info}**, the HOD is **{value}**."
            ]
        }
        
        if key_type in templates:
            return random.choice(templates[key_type])
        return value

    def _load_corpus(self):
        """Load corpus from JSONL file"""
        documents = []
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                documents.append(doc)
        return documents
    
    def _test_ollama(self):
        """Test Ollama connection"""
        try:
            response = requests.post(
                self.ollama_url,
                json={"model": self.ollama_model, "prompt": "test", "stream": False},
                timeout=90
            )
            if response.status_code == 200:
                print(f"✓ Connected to Ollama: {self.ollama_model}")
                return True
        except Exception as e:
            print(f"⚠ Ollama not available: {e}")
        return False
    
    def _is_college_related(self, query):
        """Check if query is related to college/education domain"""
        query_lower = query.lower()
        
        # College-related keywords
        college_keywords = [
            'college', 'tkrcet', 'university', 'campus', 'admission', 'course', 'department',
            'principal', 'hod', 'dean', 'faculty', 'professor', 'teacher', 'staff',
            'fee', 'placement', 'placed', 'job', 'recruit', 'company', 'companies', 'package',
            'hostel', 'library', 'lab', 'facility', 'infrastructure', 'alumni',
            'exam', 'semester', 'academic', 'student', 'class', 'lecture', 'timing',
            'branch', 'cse', 'ece', 'eee', 'mech', 'civil', 'mba', 'btech', 'mtech',
            'naac', 'nba', 'aicte', 'jntuh', 'affiliation', 'accreditation',
            'transport', 'canteen', 'sports', 'club', 'event', 'fest', 'workshop',
            'scholarship', 'eligibility', 'criteria', 'counseling', 'eapcet',
            'ncc', 'nss', 'cadet', 'service scheme',
            'syllabus', 'curriculum', 'subjects', 'exam'
        ]
        
        # Check if any college keyword is present
        if any(keyword in query_lower for keyword in college_keywords):
            return True
        
        # Check for question patterns that are likely college-related
        college_patterns = [
            'where is', 'when was', 'who is', 'what are', 'how to',
            'tell me about', 'information about', 'details about'
        ]
        
        # If it's a question pattern, check if it could be college-related
        if any(pattern in query_lower for pattern in college_patterns):
            # Reject obvious non-college topics
            non_college_keywords = [
                'formula', 'equation', 'calculate', 'solve', 'math', 'physics',
                'chemistry', 'biology', 'theorem', 'proof', '^', '=', '+', '-', '*', '/'
            ]
            if any(keyword in query_lower for keyword in non_college_keywords):
                return False
            # If it's a question but no clear non-college indicators, allow it
            return True
        
        return False
    
    def _build_kb_index(self):
        """Pre-compute embeddings for all KB entries for semantic matching"""
        import numpy as np
        
        self.kb_entries = []
        self.kb_embeddings = []
        
        def flatten_kb(data, category="", parent_key=""):
            """Recursively flatten nested KB structure"""
            for key, value in data.items():
                current_key = f"{parent_key}.{key}" if parent_key else key
                
                if isinstance(value, dict):
                    # Recursively flatten nested dicts
                    flatten_kb(value, category or key, current_key)
                elif isinstance(value, list):
                    # Handle lists (like colleges list)
                    text_value = ", ".join(str(item) for item in value)
                    search_text = f"{category} {key} {text_value}"
                    self.kb_entries.append({
                        'category': category or key,
                        'key': key,
                        'value': text_value,
                        'search_text': search_text
                    })
                elif isinstance(value, str):
                    # Create searchable text combining category, key, and value
                    search_text = f"{category} {key} {value}"
                    self.kb_entries.append({
                        'category': category or key,
                        'key': key,
                        'value': value,
                        'search_text': search_text
                    })
        
        # Flatten the KB
        flatten_kb(KNOWLEDGE_BASE)
        
        # Compute embeddings for all entries
        search_texts = [entry['search_text'] for entry in self.kb_entries]
        self.kb_embeddings = self.kb_encoder.encode(search_texts, show_progress_bar=False)
        self.kb_embeddings = np.array(self.kb_embeddings)
    
    def _check_knowledge_base(self, query):
        """KB matching with keyword fallback + semantic matching"""
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        if len(self.kb_entries) == 0:
            return None
        
        query_lower = query.lower()
        
        # ============================================================
        # EXPLICIT KEYWORD MATCHING (guaranteed instant responses)
        # ============================================================
        
        # Log check
        if any(keyword in query_lower for keyword in ['principal', 'hod', 'dean', 'courses', 'timings', 'fees']):
            print(f"  ⚡ [Fast Track] Keywords found in: '{query}'")

        # Principal
        if 'principal' in query_lower and 'vice' not in query_lower:
            return self._random_response(KNOWLEDGE_BASE['personnel']['principal'], "principal")

        # Vice Principal
        if 'vice principal' in query_lower:
            return self._random_response(KNOWLEDGE_BASE['personnel']['vice_principal'], "vice_principal")

        # Secretary / Chairman
        if 'secretary' in query_lower:
            return self._random_response(KNOWLEDGE_BASE['personnel']['secretary'], "secretary")
        if 'chairman' in query_lower:
            return self._random_response(KNOWLEDGE_BASE['society']['chairman'], "chairman")

        # Dean Academics
        if 'dean' in query_lower and 'academic' in query_lower:
             return self._random_response(KNOWLEDGE_BASE['personnel']['dean_academics'], "dean")

        # HODs
        if 'hod' in query_lower:
            # Check for specific departments
            deps = {
                'cse': 'CSE', 'aiml': 'CSE-AIML', 'ds': 'CSE-DS', 'data science': 'CSE-DS',
                'csd': 'CSE-DS', 'ai': 'CSE-AIML', 'ml': 'CSE-AIML',
                'ece': 'ECE', 'eee': 'EEE', 'it': 'IT', 'mech': 'Mechanical', 'civil': 'Civil', 'mba': 'MBA'
            }
            found_dept = False
            for key, label in deps.items():
                if key in query_lower:
                    hod_name = KNOWLEDGE_BASE['personnel']['hod'].get(key if key in ['cse', 'ece', 'eee', 'it', 'mech', 'civil', 'mba'] else 'cse-'+key if 'cse' not in key else key)
                    # Fallback for mapping
                    if not hod_name:
                         # Try direct access if key matches schema
                         hod_name = KNOWLEDGE_BASE['personnel']['hod'].get(key)
                    
                    if hod_name:
                        return self._random_response(hod_name, "hod", label)
                        found_dept = True
                        break
            
            if not found_dept:
                return "Which department's HOD are you looking for? (e.g., CSE, ECE, Mechanical)"

        # Courses / Branches
        if any(word in query_lower for word in ['courses', 'branches', 'groups', 'programmes', 'programs']):
            ug = ', '.join(KNOWLEDGE_BASE['courses']['ug'])
            pg = ', '.join(KNOWLEDGE_BASE['courses']['pg'])
            return f"**Courses Offered:**\n\n🎓 **B.Tech:** {ug}\n\n🎓 **M.Tech/MBA:** {pg}"

        # Timings
        if any(word in query_lower for word in ['timing', 'timings', 'hours', 'schedule', 'time table', 'timetable']):
            lunch = KNOWLEDGE_BASE['timings'].get('lunch_break', '')
            hours = KNOWLEDGE_BASE['timings']['working_hours']
            return f"**College Timings:**\n\n🕐 {hours}\n\n**Lunch Break:** {lunch}"
        
        # Address/Location
        if any(word in query_lower for word in ['address', 'location', 'where is', 'where are']):
            h = KNOWLEDGE_BASE['history']
            return f"**TKRCET Location:**\n\n📍 {h['location']}\n\n**Established:** {h['established']}\n**Affiliation:** {h['affiliation']}\n**Status:** {h['status']}\n**Campus Size:** {h['campus_size']}"
        
        # Fee Structure (comprehensive)
        if 'fee' in query_lower and ('structure' in query_lower or 'how much' in query_lower or 'cost' in query_lower):
            f = KNOWLEDGE_BASE['fees']
            return f"**Fee Structure (Approximate):**\n\n• **B.Tech:** {f['btech']}\n• **M.Tech:** {f['mtech']}\n• **MBA:** {f['mba']}\n\n• **Hostel:** {f['hostel']}\n• **Transport:** {f['transport']}\n\n_{f['note']}_"
        
        # Fee Payment
        if 'fee' in query_lower and ('pay' in query_lower or 'payment' in query_lower):
            return "**Fee Payment:**\n\nFees can be paid at the Accounts Department in the Administrative Block. Payment modes include:\n• Cash\n• Demand Draft\n• Online Transfer\n\nFor detailed payment procedures, please contact the Accounts Department or visit the college office."
        
        # Transport
        if any(word in query_lower for word in ['transport', 'bus', 'buses', 'route']):
            t = KNOWLEDGE_BASE['facilities']['transport']
            return f"**College Transport:**\n\n{t['details']}\n\n**Routes:** {t['routes']}\n\n{t['contact']}"
        
        # Canteen
        if 'canteen' in query_lower or 'food' in query_lower:
            c = KNOWLEDGE_BASE['facilities']['canteen']
            return f"**{c['name']}**\n\n{c['description']}\n\n**Menu:** {c['menu']}\n**Timings:** {c['timings']}"
        
        # ============================================================
        # SEMANTIC MATCHING (for other queries)
        # ============================================================
        
        # Encode query
        query_embedding = self.kb_encoder.encode([query], show_progress_bar=False)
        
        # Compute cosine similarities
        similarities = cosine_similarity(query_embedding, self.kb_embeddings)[0]
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        # Confidence threshold
        CONFIDENCE_THRESHOLD = 0.75  # Raised from 0.55 to reduce false KB matches
        
        if best_score < CONFIDENCE_THRESHOLD:
            return None  # Let RAG handle it
        
        # Get matched entry
        matched_entry = self.kb_entries[best_idx]
        category = matched_entry['category']
        key = matched_entry['key']
        value = matched_entry['value']
        
        # Format response based on category
        if category == 'personnel':
            if key == 'principal':
                return f"The Principal of TKRCET is {value}."
            elif key == 'vice_principal':
                return f"The Vice Principal is {value}."
            elif key == 'secretary':
                return f"The Secretary of TKRCET is {value}."
            elif key == 'chairman':
                return f"The Chairman of TKRCET is {value}."
            elif key == 'dean_academics':
                return f"The Dean of Academics is {value}."
            elif 'hod' in key:
                dept = key.split('.')[-1] if '.' in key else key
                return f"The HOD of {dept.upper()} is {value}."
            else:
                return value
        
        elif category == 'timings':
            if key == 'working_hours':
                lunch = KNOWLEDGE_BASE['timings'].get('lunch_break', '')
                return f"College timings: {value}. Lunch break: {lunch}."
            else:
                return value
        
        elif category == 'history' or key in ['location', 'established', 'affiliation', 'status', 'campus_size']:
            # Handle location/address queries
            h = KNOWLEDGE_BASE['history']
            if key == 'location' or 'address' in query.lower() or 'where' in query.lower():
                return f"**TKRCET Location:**\n\n📍 {h['location']}\n\n**Established:** {h['established']}\n**Affiliation:** {h['affiliation']}\n**Status:** {h['status']}\n**Campus Size:** {h['campus_size']}"
            elif key == 'established':
                return f"TKRCET was established in **{value}**."
            elif key == 'affiliation':
                return f"TKRCET is affiliated to **{value}**."
            elif key == 'status':
                return f"TKRCET has **{value}** status."
            else:
                return value
        
        elif category == 'transport' or 'transport' in key:
            t = KNOWLEDGE_BASE['facilities']['transport']
            return f"**College Transport:**\n{t['details']}\n\n**Routes:** {t['routes']}\n\n{t['contact']}"
        
        elif category == 'canteen' or 'canteen' in key:
            c = KNOWLEDGE_BASE['facilities']['canteen']
            return f"**{c['name']}**\n\n{c['description']}\n\n**Menu:** {c['menu']}\n**Timings:** {c['timings']}"
        
        elif category == 'campus_life' or key == 'events' or key == 'clubs':
            cl = KNOWLEDGE_BASE['activities']['campus_life']
            return f"**Campus Life at TKRCET**\n\n{cl['overview']}\n\n**Events:** {cl['events']}\n**Clubs:** {cl['clubs']}\n\n{cl['environment']}"
        
        elif category == 'ncc':
            ncc = KNOWLEDGE_BASE['activities']['ncc']
            return f"**{ncc['name']}**\n\n{ncc['description']}\n\n**Benefits:** {ncc['benefits']}"
        
        elif category == 'nss':
            nss = KNOWLEDGE_BASE['activities']['nss']
            return f"**{nss['name']}**\n\n{nss['description']}\n\n**Motto:** \"{nss['motto']}\""
        
        elif category == 'society' or key == 'colleges':
            s = KNOWLEDGE_BASE['society']
            colleges = "\n".join([f"{i+1}. {c}" for i, c in enumerate(s['colleges'])])
            return f"The **{s['name']}** manages the following institutions:\n\n{colleges}"
        
        elif category == 'courses':
            ug = ', '.join(KNOWLEDGE_BASE['courses']['ug'])
            pg = ', '.join(KNOWLEDGE_BASE['courses']['pg'])
            return f"TKRCET offers {KNOWLEDGE_BASE['courses']['total']}.\n\nUG Programs: {ug}\n\nPG Programs: {pg}"
        
        elif category == 'fees' or 'fee' in key:
            f = KNOWLEDGE_BASE['fees']
            return f"**Fee Structure (Approximate):**\n\n• **B.Tech:** {f['btech']}\n• **M.Tech:** {f['mtech']}\n• **MBA:** {f['mba']}\n\n• **Hostel:** {f['hostel']}\n• **Transport:** {f['transport']}\n\n_{f['note']}_"
        
        elif category == 'exam_info':
            # Return the specific exam info value
            return value
        
        else:
            # Default: return the value as-is
            return value


    def _hybrid_retrieve(self, query, top_k=5):
        """Retrieve using ChromaDB (Semantic Search) with relevance filtering"""
        if not self.collection:
           print("⚠ Database not initialized, skipping retrieval")
           return []
           
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )
            
            docs = []
            
            # Chroma returns dict of lists, we need to restructure
            # results['documents'][0] is list of chunks
            # results['metadatas'][0] is list of metadata dicts
            # results['distances'][0] is list of distance scores (lower = more similar)
            
            if not results['documents']:
                return []
            
            # Get distances if available for filtering
            distances = results.get('distances', [[]])[0] if results.get('distances') else []
                
            for i in range(len(results['documents'][0])):
                content = results['documents'][0][i]
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                distance = distances[i] if i < len(distances) else 0
                
                # Filter out low-relevance documents
                # Distance threshold: 0.0 = perfect match, 2.0 = completely different
                # Keep documents with distance < 1.0 for balanced precision/recall
                if distance > 1.0:
                    print(f"  [Filtering] Skipping low-relevance doc (distance: {distance:.3f})")
                    continue
                
                docs.append({
                    "contents": content,
                    "metadata": metadata,
                    "relevance_score": 1 - distance  # Convert distance to similarity score
                })
            
            # If we filtered out everything, return top 2 anyway (better than nothing)
            if not docs and len(results['documents'][0]) > 0:
                print("  [Warning] All docs filtered, returning top 2 anyway")
                for i in range(min(2, len(results['documents'][0]))):
                    content = results['documents'][0][i]
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    distance = distances[i] if i < len(distances) else 0
                    docs.append({
                        "contents": content,
                        "metadata": metadata,
                        "relevance_score": 1 - distance
                    })
            
            return docs
            
        except Exception as e:
            print(f"⚠ ChromaDB retrieval error: {e}")
            return []
    
    # Topic-to-URL mapping using real TKRCET website pages
    TOPIC_LINKS = {
        'admission': {'title': 'Admissions', 'url': 'https://tkrcet.ac.in/admission-procedure/'},
        'fee': {'title': 'Fee Structure', 'url': 'https://tkrcet.ac.in/fee-structure/'},
        'placement': {'title': 'Placements', 'url': 'https://tkrcet.ac.in/placements/'},
        'syllabus': {'title': 'Syllabus', 'url': 'https://tkrcet.ac.in/syllabus/'},
        'principal': {'title': 'Principal', 'url': 'https://tkrcet.ac.in/principal/'},
        'chairman': {'title': "Chairman's Message", 'url': 'https://tkrcet.ac.in/chairmans-message/'},
        'cse': {'title': 'CSE Department', 'url': 'https://tkrcet.ac.in/computer-science-engineering/'},
        'ece': {'title': 'ECE Department', 'url': 'https://tkrcet.ac.in/electronics-communication-engineering/'},
        'eee': {'title': 'EEE Department', 'url': 'https://tkrcet.ac.in/electrical-electronics-engineering/'},
        'it': {'title': 'IT Department', 'url': 'https://tkrcet.ac.in/information-technology/'},
        'mech': {'title': 'Mechanical Dept', 'url': 'https://tkrcet.ac.in/mechanical-engineering/'},
        'civil': {'title': 'Civil Dept', 'url': 'https://tkrcet.ac.in/civil-engineering/'},
        'aiml': {'title': 'CSE-AIML Dept', 'url': 'https://tkrcet.ac.in/cse-artificial-intelligence-machine-learning/'},
        'hostel': {'title': 'Campus Life', 'url': 'https://tkrcet.ac.in/about-the-campus/'},
        'campus': {'title': 'About Campus', 'url': 'https://tkrcet.ac.in/about-the-campus/'},
        'library': {'title': 'Library', 'url': 'https://tkrcet.ac.in/library/'},
        'exam': {'title': 'Academics', 'url': 'https://tkrcet.ac.in/academic-regulations/'},
        'calendar': {'title': 'Academic Calendar', 'url': 'https://tkrcet.ac.in/academic-calendars/'},
        'ncc': {'title': 'Campus Life', 'url': 'https://tkrcet.ac.in/about-the-campus/'},
        'nss': {'title': 'Campus Life', 'url': 'https://tkrcet.ac.in/about-the-campus/'},
        'event': {'title': 'TKRCET Home', 'url': 'https://tkrcet.ac.in/'},
        'fest': {'title': 'TKRCET Home', 'url': 'https://tkrcet.ac.in/'},
        'naac': {'title': 'NAAC', 'url': 'https://tkrcet.ac.in/naac-2/'},
        'alumni': {'title': 'Alumni', 'url': 'https://tkrcet.ac.in/alumni-sub-domain/'},
        'transport': {'title': 'About Campus', 'url': 'https://tkrcet.ac.in/about-the-campus/'},
        'mba': {'title': 'MBA Department', 'url': 'https://tkrcet.ac.in/mba/'},
    }

    def _get_topic_links(self, query):
        """Get relevant website links based on query topic keywords"""
        query_lower = query.lower()
        links = []
        seen_urls = set()
        
        for keyword, link_info in self.TOPIC_LINKS.items():
            if keyword in query_lower and link_info['url'] not in seen_urls:
                seen_urls.add(link_info['url'])
                links.append(link_info)
        
        # Always add the main website as a fallback if no topic matched
        if not links:
            links.append({'title': 'TKRCET Website', 'url': 'https://tkrcet.ac.in/'})
        
        return links[:3]  # Max 3 links

    def _extract_relevant_links(self, docs, query=''):
        """Extract URLs from documents, falling back to topic-based links"""
        links = []
        seen_urls = set()
        
        for doc in docs[:5]:
            url = doc.get('metadata', {}).get('url', '')
            source = doc.get('metadata', {}).get('source', '')
            link = source if source and source.startswith('http') else url
            
            if link and link not in seen_urls and link.startswith('http'):
                seen_urls.add(link)
                links.append(link)
        
        # If no document URLs found, use topic-based links
        if not links and query:
            topic_links = self._get_topic_links(query)
            return topic_links  # Returns list of {title, url} dicts
        
        return links[:3]
    
    def _generate_response(self, query, docs, language='en', stream=False):
        """Generate response using Ollama with retrieved context
        
        Args:
            query: User query
            docs: Retrieved documents
            language: Response language ('en', 'hi', 'te')
            stream: If True, yields chunks. If False, returns complete response.
        """
        
        # Extract links first for Quick Links section
        links = self._extract_relevant_links(docs, query=query)
        
        # Build context from retrieved documents (Increased snippet length for better detail)
        context = "\n\n".join([f"• {doc['contents'][:1000]}" for doc in docs[:5]])
        
        # Build KB context
        kb_context = self._format_kb_context()
        
        lang_instruction = ""
        if language == 'hi':
            lang_instruction = "IMPORTANT: Answer the student's question in HINDI (हिंदी). Transliterate technical terms if needed."
        elif language == 'te':
            lang_instruction = "IMPORTANT: Answer the student's question in TELUGU (తెలుగు). Transliterate technical terms if needed."
        else:
            lang_instruction = "Answer in English."

        prompt = f"""Hey! You're the friendly TKRCET College Buddy 😊 - think of yourself as a helpful senior student who knows everything about the college.
{lang_instruction}

YOUR PERSONALITY:
- Be warm, friendly, and conversational (like chatting with a friend)
- Use casual language but stay professional
- Show enthusiasm about TKRCET!
- Be understanding of typos and unclear questions

GUIDELINES:
- **Context is Key**: Always assume questions are about TKRCET. "What's the process?" = "TKRCET admission process"
- **Be Helpful**: If you don't have exact info, guide them to the right resource or office
- **Keep it Natural**: Avoid robotic responses - talk like a real person!

FORMATTING:
- Use **bold** for important names and numbers
- Use bullet points for lists
- **Be comprehensive yet concise**: Provide a complete answer but avoid fluff.
- If the question is complex, provide a step-by-step guide.

What I Know About TKRCET:
{kb_context}

Relevant Info:
{context}

Student's Question: {query}

Your Friendly Response:"""
        
        # Build Quick Links section (appears at top)
        quick_links_section = ""
        if links:
            quick_links_section = "📌 **Quick Links:**\n"
            for link in links:
                # Handle both dict format (topic links) and string format (document links)
                if isinstance(link, dict):
                    title = link['title']
                    url = link['url']
                elif 'tkrcet' in link.lower():
                    title = "TKRCET Official Page"
                    url = link
                else:
                    title = "Related Resource"
                    url = link
                quick_links_section += f"• [{title}]({url})\n"
            quick_links_section += "\n"
        
        # Build source links footer
        source_links_section = ""
        if links:
            source_links_section = "\n\n📚 **Source Links:**\n"
            for link in links:
                if isinstance(link, dict):
                    source_links_section += f"• [{link['title']}]({link['url']})\n"
                else:
                    source_links_section += f"• {link}\n"
        
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": stream,  # Enable streaming if requested
                    "options": {
                        "temperature": 0.7,  # Natural conversation
                        "top_k": 40,         # Balanced creativity
                        "top_p": 0.9,        # Balanced coherence
                        "num_predict": 512,  # Increased for complete responses (was 60)
                        "num_ctx": 2048      # Increased for better memory (was 256)
                    }
                },
                timeout=120,  # Increased to 120s to prevent timeouts on complex/slow queries
                stream=stream  # Enable streaming in requests library
            )
            
            if stream:
                # Streaming mode: yield chunks as they arrive
                # First, yield the quick links section
                if quick_links_section:
                    yield quick_links_section
                
                # Then yield the streamed response
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk_data = json.loads(line)
                            if 'response' in chunk_data:
                                yield chunk_data['response']
                            
                            # Check if done
                            if chunk_data.get('done', False):
                                # Yield source links at the end
                                if source_links_section:
                                    yield source_links_section
                                break
                        except json.JSONDecodeError:
                            continue
            else:
                # Non-streaming mode (original behavior)
                if response.status_code == 200:
                    answer = response.json().get('response', '').strip()
                    if answer:
                        # Build final response: Quick Links + Answer + Source Links
                        final_response = quick_links_section + answer + source_links_section
                        return final_response
        except Exception as e:
            print(f"⚠ Ollama error: {e}")
            # Fallback to document snippets with links
            fallback = quick_links_section + f"Here's what I found:\n\n{context}" + source_links_section
            if stream:
                yield fallback
            else:
                return fallback
        
        # Fallback for non-streaming if nothing was returned
        if not stream:
            fallback = quick_links_section + f"Here's what I found:\n\n{context}" + source_links_section
            return fallback
    
    def _format_kb_context(self):
        """Format knowledge base as context"""
        kb = KNOWLEDGE_BASE
        lines = []
        
        lines.append(f"Principal: {kb['personnel']['principal']}")
        lines.append(f"Vice Principal: {kb['personnel']['vice_principal']}")
        lines.append(f"Timings: {kb['timings']['working_hours']}")
        lines.append(f"Founded: {kb['history']['established']}")
        lines.append(f"Affiliation: {kb['history']['affiliation']}")
        
        if 'statistics' in kb:
            stats = kb['statistics']
            lines.append(f"\nFAST FACTS:")
            lines.append(f"• Total Students: {stats['total_students']}")
            lines.append(f"• Placed Students: {stats['placed_students']} (Rate: {stats['placement_rate']})")
            lines.append(f"• Top Recruiters: {stats['top_recruiters']}")
        
        return "\n".join(lines)
    
    def __call__(self, query, language='en', stream=False, return_dict=False):
        """Main entry point for queries
        
        Args:
            query: User query string
            language: Response language ('en', 'hi', 'te')
            stream: If True, yields response chunks. If False, returns complete response.
            return_dict: If True (and not streaming), returns dict with response and metadata.
        """
        query = query.strip()
        if not query:
            if stream:
                yield "Please enter a question."
                return
            return {"response": "Please enter a question.", "source": "System"} if return_dict else "Please enter a question."
        
        # Greetings (Flexible Regex Matching)
        import re
        greetings_pattern = r"^(hi|hello|hey|greetings|how are you|how r u|how are u|whats up|what's up|how do you do|good morning|good afternoon|good evening)[\s\?\!\.]*$"
        
        if re.match(greetings_pattern, query.lower()):
            greeting_response = "Hello! I'm your TKRCET College Buddy! 😊 How can I help you today!"
            if stream:
                yield {"type": "metadata", "source": "Greeting"}
                yield greeting_response
                return
            return {"response": greeting_response, "source": "Greeting"} if return_dict else greeting_response
        
        # Check cache first (for exact query matches) - only for non-streaming
        if not stream:
            query_key = query.lower().strip()
            if query_key in self.response_cache:
                print("  [Cache Hit] Returning cached response")
                return {"response": self.response_cache[query_key], "source": "Cache"} if return_dict else self.response_cache[query_key]
        
        # Check knowledge base first
        kb_answer = self._check_knowledge_base(query)
        if kb_answer:
            # Append topic-based navigation links to KB answers
            topic_links = self._get_topic_links(query)
            if topic_links:
                links_section = "\n\n📌 **Quick Links:**\n"
                for tl in topic_links:
                    links_section += f"• [{tl['title']}]({tl['url']})\n"
                kb_answer = kb_answer + links_section

            if not stream:
                # Cache knowledge base answers
                query_key = query.lower().strip()
                self.response_cache[query_key] = kb_answer
                # Limit cache size to 50 entries
                if len(self.response_cache) > 50:
                    # Remove oldest entry (FIFO)
                    self.response_cache.pop(next(iter(self.response_cache)))
            
            if stream:
                yield {"type": "metadata", "source": "Knowledge Base"}
                yield kb_answer
                return
            return {"response": kb_answer, "source": "Knowledge Base"} if return_dict else kb_answer
        
        # Retrieve relevant documents
        docs = self._hybrid_retrieve(query, top_k=3)
        
        # Calculate real confidence from retrieval scores
        if docs:
            avg_score = sum(d.get('relevance_score', 0) for d in docs) / len(docs)
            confidence = max(0, min(100, int(avg_score * 100)))
        else:
            confidence = 0
        
        # Generate response
        if stream:
            yield {"type": "metadata", "source": "RAG", "confidence": confidence}
            # Streaming mode: yield chunks
            for chunk in self._generate_response(query, docs, language, stream=True):
                yield chunk
        else:
            # Non-streaming mode: return complete response
            response = self._generate_response(query, docs, language, stream=False)
            if return_dict:
                return {"response": response, "source": "RAG", "confidence": confidence}
            return response


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("ULTRARAG SYSTEM TEST")
    print("=" * 70 + "\n")
    
    rag = UltraRAGSystem()
    
    test_queries = [
        "hi",
        "who is the principal?",
        "who is the HOD of CSE?",
        "college timings?",
        "when was college established?",
        "what are the facilities?",
        "tell me about the canteen",
        "how is the campus life?",
    ]
    
    for query in test_queries:
        print(f"Q: {query}")
        answer = rag(query)
        print(f"A: {answer}\n")
