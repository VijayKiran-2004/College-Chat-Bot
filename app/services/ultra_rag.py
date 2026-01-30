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
    }
}


class UltraRAGSystem:
    """
    UltraRAG-based RAG system for college-buddy chatbot
    Uses MCP architecture with FAISS + BM25 hybrid retrieval
    """
    
    def __init__(
        self,
        corpus_path='app/database/vectordb/corpus_ultrarag.jsonl',
        ollama_model=None,
        ollama_url=None,
    ):
        # Configuration from arguments or environment variables
        self.ollama_model = ollama_model or os.environ.get('OLLAMA_MODEL', 'gemma2:2b')
        self.ollama_url = ollama_url or os.environ.get('OLLAMA_URL', 'http://localhost:11434/api/generate')
        
        
        print("Initializing UltraRAG System...")
        
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
            self.chroma_client = chromadb.PersistentClient(path='app/database/vectordb/chroma')
            
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
                timeout=30
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
    
    def _check_knowledge_base(self, query):
        """Check if query can be answered from knowledge base"""
        query_lower = query.lower()
        
        # Personnel queries
        if 'principal' in query_lower and 'vice' not in query_lower:
            return f"The Principal of TKRCET is {KNOWLEDGE_BASE['personnel']['principal']}."
        
        if 'vice principal' in query_lower:
            return f"The Vice Principal is {KNOWLEDGE_BASE['personnel']['vice_principal']}."
        
        if 'secretary' in query_lower:
            return f"The Secretary of TKRCET is {KNOWLEDGE_BASE['personnel']['secretary']}."
        
        if 'chairman' in query_lower:
            return f"The Chairman of TKRCET is {KNOWLEDGE_BASE['personnel']['chairman']}."
        
        if 'founder' in query_lower:
            return f"The Founder of TKRCET is {KNOWLEDGE_BASE['personnel']['chairman']} (Founder Chairman of TKR Educational Society)."
        
        if 'dean' in query_lower:
            return f"The Dean of Academics is {KNOWLEDGE_BASE['personnel']['dean_academics']}."
        
        if 'hod' in query_lower or 'head of department' in query_lower:
            for dept, hod in KNOWLEDGE_BASE['personnel']['hod'].items():
                if dept in query_lower:
                    return f"The HOD of {dept.upper()} is {hod}."
        
        # Timings
        if 'timing' in query_lower or 'hours' in query_lower or 'time' in query_lower:
            return f"College timings: {KNOWLEDGE_BASE['timings']['working_hours']}. Lunch break: {KNOWLEDGE_BASE['timings']['lunch_break']}."
        
        # History & Location
        if 'established' in query_lower or 'founded' in query_lower or 'started' in query_lower:
            return f"TKRCET was established in {KNOWLEDGE_BASE['history']['established']} on a {KNOWLEDGE_BASE['history']['campus_size']} campus in {KNOWLEDGE_BASE['history']['location']}."
        
        if 'affiliation' in query_lower or 'affiliated' in query_lower:
            return f"TKRCET is affiliated to {KNOWLEDGE_BASE['history']['affiliation']}."
        
        if 'location' in query_lower or 'address' in query_lower or 'where' in query_lower:
            return f"TKRCET is located at {KNOWLEDGE_BASE['history']['location']}."
        
        # Admissions
        if 'admission' in query_lower or 'apply' in query_lower or 'join' in query_lower:
            return f"{KNOWLEDGE_BASE['admissions']['process']} Eligibility: {KNOWLEDGE_BASE['admissions']['eligibility']} {KNOWLEDGE_BASE['admissions']['contact']}"
        
        # Courses
        if 'course' in query_lower or 'program' in query_lower or 'branch' in query_lower or 'department' in query_lower:
            ug = ', '.join(KNOWLEDGE_BASE['courses']['ug'])
            pg = ', '.join(KNOWLEDGE_BASE['courses']['pg'])
            return f"TKRCET offers {KNOWLEDGE_BASE['courses']['total']}.\n\nUG Programs: {ug}\n\nPG Programs: {pg}"
        
        if 'transport' in query_lower or 'bus' in query_lower:
            t = KNOWLEDGE_BASE['facilities']['transport']
            return f"**College Transport:**\n{t['details']}\n\n**Routes:** {t['routes']}\n\n{t['contact']}"

        if 'canteen' in query_lower or 'food' in query_lower or 'cafeteria' in query_lower:
            c = KNOWLEDGE_BASE['facilities']['canteen']
            return f"**{c['name']}**\n\n{c['description']}\n\n**Menu:** {c['menu']}\n**Timings:** {c['timings']}"

        # Fees
        if 'fee' in query_lower or 'cost' in query_lower or 'payment' in query_lower:
            f = KNOWLEDGE_BASE['fees']
            return f"**Fee Structure (Approximate):**\n\n• **B.Tech:** {f['btech']}\n• **M.Tech:** {f['mtech']}\n• **MBA:** {f['mba']}\n\n• **Hostel:** {f['hostel']}\n• **Transport:** {f['transport']}\n\n_{f['note']}_"

        # NCC
        if 'ncc' in query_lower or 'cadet' in query_lower:
            ncc = KNOWLEDGE_BASE['activities']['ncc']
            return f"**{ncc['name']}**\n\n{ncc['description']}\n\n**Benefits:** {ncc['benefits']}"

        # NSS
        if 'nss' in query_lower:
            nss = KNOWLEDGE_BASE['activities']['nss']
            return f"**{nss['name']}**\n\n{nss['description']}\n\n**Motto:** \"{nss['motto']}\""

        # Campus Life
        if 'life' in query_lower or 'culture' in query_lower or 'events' in query_lower or 'fests' in query_lower or 'clubs' in query_lower:
             cl = KNOWLEDGE_BASE['activities']['campus_life']
             return f"**Campus Life at TKRCET**\n\n{cl['overview']}\n\n**Events:** {cl['events']}\n**Clubs:** {cl['clubs']}\n\n{cl['environment']}"

        # Facilities
        if 'facilit' in query_lower or 'infrastructure' in query_lower or 'amenities' in query_lower:
            return f"{KNOWLEDGE_BASE['facilities']['main']}\n\nSpecial Features: {KNOWLEDGE_BASE['facilities']['special']}"
        
        # Accreditation
        if 'naac' in query_lower or 'nba' in query_lower or 'accredit' in query_lower or 'approved' in query_lower:
            return f"TKRCET is {KNOWLEDGE_BASE['accreditation']['naac']} accredited, {KNOWLEDGE_BASE['accreditation']['nba']}, and {KNOWLEDGE_BASE['accreditation']['approvals']}."
        
        # Autonomous Status
        if 'autonomous' in query_lower:
            return f"Yes, TKRCET is an **Autonomous** institution affiliated to JNTUH. It has UGC confirmation for its autonomous status, allowing academic freedom in curriculum and evaluation."

        # Society / Colleges
        if 'society' in query_lower or 'colleges' in query_lower or 'institutions' in query_lower:
            s = KNOWLEDGE_BASE['society']
            colleges = "\n".join([f"{i+1}. {c}" for i, c in enumerate(s['colleges'])])
            return f"The **{s['name']}** manages the following institutions:\n\n{colleges}"

        # Full Form / Abbreviation
        if 'full form' in query_lower or 'stand for' in query_lower or 'meaning of tkr' in query_lower:
            if 'tkr' in query_lower:
                return f"**TKR** stands for **{KNOWLEDGE_BASE['society']['full_form']}** (named after the founder and chairman, Sri. Teegala Krishna Reddy)."
        
        return None
    
    def _hybrid_retrieve(self, query, top_k=5):
        """Retrieve using ChromaDB (Semantic Search)"""
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
            
            if not results['documents']:
                return []
                
            for i in range(len(results['documents'][0])):
                content = results['documents'][0][i]
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                
                docs.append({
                    "contents": content,
                    "metadata": metadata
                })
            
            return docs
            
        except Exception as e:
            print(f"⚠ ChromaDB retrieval error: {e}")
            return []
    
    def _extract_relevant_links(self, docs):
        """Extract unique, relevant URLs from retrieved documents"""
        links = []
        seen_urls = set()
        
        for doc in docs[:5]:  # Check top 5 docs
            url = doc.get('metadata', {}).get('url', '')
            source = doc.get('metadata', {}).get('source', '')
            
            # Prefer source over url field
            link = source if source and source.startswith('http') else url
            
            if link and link not in seen_urls and link.startswith('http'):
                seen_urls.add(link)
                links.append(link)
        
        return links[:3]  # Return max 3 links
    
    def _generate_response(self, query, docs, language='en'):
        """Generate response using Ollama with retrieved context"""
        
        # Build context from retrieved documents
        context = "\n\n".join([f"• {doc['contents'][:400]}" for doc in docs[:3]])
        
        # Build KB context
        kb_context = self._format_kb_context()
        
        lang_instruction = ""
        if language == 'hi':
            lang_instruction = "IMPORTANT: Answer the student's question in HINDI (हिंदी). Transliterate technical terms if needed."
        elif language == 'te':
            lang_instruction = "IMPORTANT: Answer the student's question in TELUGU (తెలుగు). Transliterate technical terms if needed."
        else:
            lang_instruction = "Answer in English."

        prompt = f"""You are the TKRCET College Assistant chatbot. You primarily answer questions about TKRCET college, but you can be helpful with general queries as well.
{lang_instruction}

GUIDELINES:
- **Contextualize Everything**: ALWAYS interpret the user's question in the context of **TKRCET College**. For example, if they ask "what is the process?", assume they mean "TKRCET Admission Process".
- **Typos**: Be tolerant of typos (e.g., "addmission", "fess").
- **Primary Focus**: Prioritize answering questions about TKRCET (admissions, courses, facilities, personnel, etc.).
- **student Data**: If the user asks about specific student data (placements, CGPA) and you don't have it in the context, suggest they ask for "student records" or "placement data" specifically so the system can look it up.

FORMATTING RULES:
- Use **bold** for key terms, names, and important numbers.
- Use bullet points for lists.
- Keep output concise and readable.

Context Information:
{kb_context}

{context}

Student Question: {query}

Your Answer:
(Provide a helpful answer based on the context.)

Student Question: {query}
 
Your Answer:
(Provide a helpful answer based on the context.)"""
        
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,  # Increased for more natural conversation
                        "top_k": 40,         # Balanced creativity
                        "top_p": 0.9,        # Balanced coherence
                        "num_predict": 250,  # Increased slightly for fuller answers
                        "num_ctx": 2048      # Restored to standard context size
                    }
                },
                timeout=60
            )
            if response.status_code == 200:
                answer = response.json().get('response', '').strip()
                if answer:
                    # Add relevant navigation links
                    links = self._extract_relevant_links(docs)
                    if links:
                        answer += "\n\n📌 Related Links:\n"
                        for link in links:
                            answer += f"• {link}\n"
                    return answer
        except Exception as e:
            print(f"⚠ Ollama error: {e}")
        
        # Fallback to document snippets with links
        fallback = f"Here's what I found:\n\n{context}"
        links = self._extract_relevant_links(docs)
        if links:
            fallback += "\n\n📌 Related Links:\n"
            for link in links:
                fallback += f"• {link}\n"
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
    
    def __call__(self, query, language='en'):
        """Main entry point for queries"""
        query = query.strip()
        if not query:
            return "Please enter a question."
        
        # Greetings
        greetings = ['hi', 'hello', 'hey', 'how are you', 'how r u', 'how are u', 'whats up', "what's up"]
        if any(greeting == query.lower() for greeting in greetings):
            return "Hello! I'm TKRCET College Assistant. How can I help you today? 😊"
        
        # Check if query is college-related
        # if not self._is_college_related(query):
        #     return "I'm sorry, I can only answer questions about TKRCET college. Please ask me about admissions, courses, facilities, timings, faculty, or other college-related topics."
        
        # Check knowledge base first
        kb_answer = self._check_knowledge_base(query)
        if kb_answer:
            return kb_answer
        
        # Retrieve relevant documents
        docs = self._hybrid_retrieve(query, top_k=3)
        
        # Generate response
        response = self._generate_response(query, docs, language)
        
        # Append Related Content Links
        links = self._extract_relevant_links(docs)
        if links:
            response += "\n\nRelated Links:"
            for link in links:
                # Basic formatting check
                if 'tkrcet' in link:
                    title = "TKRCET Page"
                else:
                    title = "Source"
                response += f"\n- [{title}]({link})"
        
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
