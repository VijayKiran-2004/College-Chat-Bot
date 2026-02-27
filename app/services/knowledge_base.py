import json
import numpy as np

class KnowledgeBase:
    """Handles fast knowledge base retrieval using exact matching and semantic fallback"""
    
    def __init__(self, kb_path, semantic_model):
        self.data = self._load_data(kb_path)
        self.kb_encoder = semantic_model
        
        self.kb_entries = []
        self.kb_embeddings = []
        
        print("Building KB semantic index...")
        self._build_kb_index()
        print("✓ KB semantic index ready")

    def _load_data(self, kb_path):
        try:
            with open(kb_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠ Could not load KB from {kb_path}: {e}")
            return {}

    def load_sql_stats(self):
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
            
            self.data['statistics'] = {
                "total_students": str(total_students),
                "placed_students": str(placed_count),
                "top_recruiters": top_companies,
                "placement_rate": f"{int((placed_count/total_students)*100)}%" if total_students > 0 else "N/A"
            }
            sql.close()
            print(f"✓ SQL Stats loaded: {total_students} students, {placed_count} placed")
            
        except Exception as e:
            print(f"⚠ Could not load SQL stats: {e}")
            self.data['statistics'] = {
                "total_students": "1600+",
                "placed_students": "Many",
                "top_recruiters": "TCS, Wipro, Infosys",
                "placement_rate": "High"
            }

    def _build_kb_index(self):
        """Pre-compute embeddings for all KB entries for semantic matching"""
        if not self.data:
            return
            
        def flatten_kb(data, category="", parent_key=""):
            for key, value in data.items():
                current_key = f"{parent_key}.{key}" if parent_key else key
                if isinstance(value, dict):
                    flatten_kb(value, category or key, current_key)
                elif isinstance(value, list):
                    text_value = ", ".join(str(item) for item in value)
                    search_text = f"{category} {key} {text_value}"
                    self.kb_entries.append({
                        'category': category or key,
                        'key': key,
                        'value': text_value,
                        'search_text': search_text
                    })
                elif isinstance(value, str):
                    search_text = f"{category} {key} {value}"
                    self.kb_entries.append({
                        'category': category or key,
                        'key': key,
                        'value': value,
                        'search_text': search_text
                    })
        
        flatten_kb(self.data)
        
        search_texts = [entry['search_text'] for entry in self.kb_entries]
        self.kb_embeddings = self.kb_encoder.encode(search_texts, show_progress_bar=False)
        self.kb_embeddings = np.array(self.kb_embeddings)

    def _random_response(self, value, key_type, extra_info=None):
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

    def check(self, query):
        """KB matching with keyword fallback + semantic matching"""
        from sklearn.metrics.pairwise import cosine_similarity
        
        if len(self.kb_entries) == 0:
            return None
            
        query_lower = query.lower()
        
        if any(keyword in query_lower for keyword in ['principal', 'hod', 'dean', 'courses', 'timings', 'fees']):
            print(f"  ⚡ [Fast Track] Keywords found in: '{query}'")

        if 'principal' in query_lower and 'vice' not in query_lower:
            return self._random_response(self.data['personnel']['principal'], "principal")

        if 'vice principal' in query_lower:
            return self._random_response(self.data['personnel']['vice_principal'], "vice_principal")

        if 'secretary' in query_lower:
            return self._random_response(self.data['personnel']['secretary'], "secretary")
        if 'chairman' in query_lower:
            return self._random_response(self.data['society']['chairman'], "chairman")

        if 'dean' in query_lower and 'academic' in query_lower:
             return self._random_response(self.data['personnel']['dean_academics'], "dean")

        if 'hod' in query_lower:
            deps = {
                'cse': 'CSE', 'aiml': 'CSE-AIML', 'ds': 'CSE-DS', 'data science': 'CSE-DS',
                'csd': 'CSE-DS', 'ai': 'CSE-AIML', 'ml': 'CSE-AIML',
                'ece': 'ECE', 'eee': 'EEE', 'it': 'IT', 'mech': 'Mechanical', 'civil': 'Civil', 'mba': 'MBA'
            }
            found_dept = False
            for key, label in deps.items():
                if key in query_lower:
                    search_key = key if key in ['cse', 'ece', 'eee', 'it', 'mech', 'civil', 'mba'] else 'cse-'+key if 'cse' not in key else key
                    hod_name = self.data['personnel']['hod'].get(search_key)
                    if not hod_name:
                         hod_name = self.data['personnel']['hod'].get(key)
                    
                    if hod_name:
                        return self._random_response(hod_name, "hod", label)
                        found_dept = True
                        break
            
            if not found_dept:
                return "Which department's HOD are you looking for? (e.g., CSE, ECE, Mechanical)"

        if any(word in query_lower for word in ['courses', 'branches', 'groups', 'programmes', 'programs']):
            ug = ', '.join(self.data['courses']['ug'])
            pg = ', '.join(self.data['courses']['pg'])
            return f"**Courses Offered:**\n\n🎓 **B.Tech:** {ug}\n\n🎓 **M.Tech/MBA:** {pg}"

        if any(word in query_lower for word in ['timing', 'timings', 'hours', 'schedule', 'time table', 'timetable']):
            lunch = self.data['timings'].get('lunch_break', '')
            hours = self.data['timings']['working_hours']
            return f"**College Timings:**\n\n🕐 {hours}\n\n**Lunch Break:** {lunch}"
        
        if any(word in query_lower for word in ['address', 'location', 'where is', 'where are']):
            h = self.data['history']
            return f"**TKRCET Location:**\n\n📍 {h['location']}\n\n**Established:** {h['established']}\n**Affiliation:** {h['affiliation']}\n**Status:** {h['status']}\n**Campus Size:** {h['campus_size']}"
        
        if 'fee' in query_lower and ('structure' in query_lower or 'how much' in query_lower or 'cost' in query_lower):
            f = self.data['fees']
            note = f.get('note', 'Fees are subject to change as per government regulations.')
            return f"**Fee Structure (Approximate):**\n\n• **B.Tech:** {f['btech']}\n• **M.Tech:** {f['mtech']}\n• **MBA:** {f['mba']}\n\n• **Hostel:** {f['hostel']}\n• **Transport:** {f['transport']}\n\n_{note}_"
        
        if 'fee' in query_lower and ('pay' in query_lower or 'payment' in query_lower):
            return "**Fee Payment:**\n\nFees can be paid at the Accounts Department in the Administrative Block. Payment modes include:\n• Cash\n• Demand Draft\n• Online Transfer\n\nFor detailed payment procedures, please contact the Accounts Department or visit the college office."
        
        if any(word in query_lower for word in ['transport', 'bus', 'buses', 'route']):
            t = self.data['facilities']['transport']
            return f"**College Transport:**\n\n{t['details']}\n\n**Routes:** {t['routes']}\n\n{t['contact']}"
        
        if 'canteen' in query_lower or 'food' in query_lower:
            c = self.data['facilities']['canteen']
            return f"**{c['name']}**\n\n{c['description']}\n\n**Menu:** {c['menu']}\n**Timings:** {c['timings']}"
        
        # Semantic Match
        query_embedding = self.kb_encoder.encode([query], show_progress_bar=False)
        similarities = cosine_similarity(query_embedding, self.kb_embeddings)[0]
        
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        CONFIDENCE_THRESHOLD = 0.75
        if best_score < CONFIDENCE_THRESHOLD:
            return None
        
        matched_entry = self.kb_entries[best_idx]
        category = matched_entry['category']
        key = matched_entry['key']
        value = matched_entry['value']
        
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
                lunch = self.data['timings'].get('lunch_break', '')
                return f"College timings: {value}. Lunch break: {lunch}."
            else:
                return value
        
        elif category == 'history' or key in ['location', 'established', 'affiliation', 'status', 'campus_size']:
            h = self.data['history']
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
            t = self.data['facilities']['transport']
            return f"**College Transport:**\n{t['details']}\n\n**Routes:** {t['routes']}\n\n{t['contact']}"
        
        elif category == 'canteen' or 'canteen' in key:
            c = self.data['facilities']['canteen']
            return f"**{c['name']}**\n\n{c['description']}\n\n**Menu:** {c['menu']}\n**Timings:** {c['timings']}"
        
        elif category == 'campus_life' or key == 'events' or key == 'clubs':
            cl = self.data['activities']['campus_life']
            return f"**Campus Life at TKRCET**\n\n{cl['overview']}\n\n**Events:** {cl['events']}\n**Clubs:** {cl['clubs']}\n\n{cl['environment']}"
        
        elif category == 'ncc':
            ncc = self.data['activities']['ncc']
            return f"**{ncc['name']}**\n\n{ncc['description']}\n\n**Benefits:** {ncc['benefits']}"
        
        elif category == 'nss':
            nss = self.data['activities']['nss']
            return f"**{nss['name']}**\n\n{nss['description']}\n\n**Motto:** \"{nss['motto']}\""
        
        elif category == 'society' or key == 'colleges':
            s = self.data['society']
            colleges = "\n".join([f"{i+1}. {c}" for i, c in enumerate(s['colleges'])])
            return f"The **{s['name']}** manages the following institutions:\n\n{colleges}"
        
        elif category == 'courses':
            ug = ', '.join(self.data['courses']['ug'])
            pg = ', '.join(self.data['courses']['pg'])
            return f"TKRCET offers {self.data['courses']['total']}.\n\nUG Programs: {ug}\n\nPG Programs: {pg}"
        
        elif category == 'fees' or 'fee' in key:
            f = self.data['fees']
            note = f.get('note', 'Fees are subject to change as per government regulations.')
            return f"**Fee Structure (Approximate):**\n\n• **B.Tech:** {f['btech']}\n• **M.Tech:** {f['mtech']}\n• **MBA:** {f['mba']}\n\n• **Hostel:** {f['hostel']}\n• **Transport:** {f['transport']}\n\n_{note}_"
        
        elif category == 'exam_info':
            return value
        else:
            return value

    def format_context(self):
        """Format knowledge base as context"""
        if not self.data:
            return ""
        
        lines = []
        lines.append(f"Principal: {self.data['personnel']['principal']}")
        lines.append(f"Vice Principal: {self.data['personnel']['vice_principal']}")
        lines.append(f"Timings: {self.data['timings']['working_hours']}")
        lines.append(f"Founded: {self.data['history']['established']}")
        lines.append(f"Affiliation: {self.data['history']['affiliation']}")
        
        if 'statistics' in self.data:
            stats = self.data['statistics']
            lines.append(f"\nFAST FACTS:")
            lines.append(f"• Total Students: {stats['total_students']}")
            lines.append(f"• Placed Students: {stats['placed_students']} (Rate: {stats['placement_rate']})")
            lines.append(f"• Top Recruiters: {stats['top_recruiters']}")
        
        return "\n".join(lines)
