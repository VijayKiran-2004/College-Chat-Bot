import json
import numpy as np


class KnowledgeBase:
    """Handles fast knowledge base retrieval
    using exact matching and semantic fallback"""

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
            with open(kb_path, "r", encoding="utf-8") as f:
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
            df_total = pd.read_sql_query(
                "SELECT COUNT(*) as count FROM students", sql.conn
            )
            total_students = df_total.iloc[0]["count"]

            # Placed students
            df_placed = pd.read_sql_query(
                "SELECT COUNT(*) as count FROM students"
                ' WHERE "COMPANY PLACED" IS NOT NULL '
                "AND \"COMPANY PLACED\" != 'Not Placed'",
                sql.conn,
            )
            placed_count = df_placed.iloc[0]["count"]

            # Top companies
            df_companies = pd.read_sql_query(
                """SELECT "COMPANY PLACED", COUNT(*) as count
                   FROM students
                   WHERE "COMPANY PLACED" IS NOT NULL
                   AND "COMPANY PLACED" != 'Not Placed'
                   GROUP BY "COMPANY PLACED"
                   ORDER BY count DESC
                   LIMIT 3""",
                sql.conn,
            )
            top_companies = ", ".join(
                [
                    f"{row['COMPANY PLACED']} ({row['count']})"
                    for _, row in df_companies.iterrows()
                ]
            )

            self.data["statistics"] = {
                "total_students": str(total_students),
                "placed_students": str(placed_count),
                "top_recruiters": top_companies,
                "placement_rate": (
                    f"{int((placed_count/total_students)*100)}%"
                    if total_students > 0
                    else "N/A"
                ),
            }
            sql.close()
            print(
                f"✓ SQL Stats loaded: {total_students} students, {placed_count} placed"
            )

        except Exception as e:
            print(f"⚠ Could not load SQL stats: {e}")
            self.data["statistics"] = {
                "total_students": "1600+",
                "placed_students": "Many",
                "top_recruiters": "TCS, Wipro, Infosys",
                "placement_rate": "High",
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
                    self.kb_entries.append(
                        {
                            "category": category or key,
                            "key": key,
                            "value": text_value,
                            "search_text": search_text,
                        }
                    )
                elif isinstance(value, str):
                    search_text = f"{category} {key} {value}"
                    self.kb_entries.append(
                        {
                            "category": category or key,
                            "key": key,
                            "value": value,
                            "search_text": search_text,
                        }
                    )

        flatten_kb(self.data)

        search_texts = [entry["search_text"] for entry in self.kb_entries]
        self.kb_embeddings = self.kb_encoder.encode(
            search_texts, show_progress_bar=False
        )
        self.kb_embeddings = np.array(self.kb_embeddings)

    def check(self, query):
        """KB matching with keyword fallback +
        semantic matching. Returns raw fact or None."""
        from sklearn.metrics.pairwise import cosine_similarity

        if len(self.kb_entries) == 0:
            return None

        query_lower = query.lower().strip()

        # 1. SPECIFIC KEYWORD MATCHES (High Priority)
        # Society / Management
        if any(word in query_lower for word in ["chairman", "chairmen", "chairman's"]):
            return f"The Chairman of TKRCET is {self.data['society']['chairman']}."

        if "secretary" in query_lower:
            return f"The Secretary of TKRCET is {self.data['society']['secretary']}."

        # Personnel / Administration
        if "principal" in query_lower and "vice" not in query_lower:
            return f"The Principal of TKRCET is {self.data['personnel']['principal']}."

        if "vice principal" in query_lower:
            return (
                "The Vice Principal of TKRCET is"
                f" {self.data['personnel']['vice_principal']}."
            )

        if "dean" in query_lower and "academic" in query_lower:
            return (
                "The Dean Academics of TKRCET is"
                f" {self.data['personnel']['dean_academics']}."
            )

        # Full college name
        if any(
            p in query_lower
            for p in [
                "full name",
                "full form",
                "what does tkrcet stand",
                "expand tkrcet",
                "college name",
            ]
        ):
            full_name = self.data.get("history", {}).get(
                "full_name",
                "Teegala Krishna Reddy College of Engineering and Technology (TKRCET)",
            )
            return f"**Full Name of College:** {full_name}"

        if "hod" in query_lower:
            deps = {
                "cse": "CSE",
                "aiml": "CSE-AIML",
                "ds": "CSE-DS",
                "data science": "CSE-DS",
                "csd": "CSE-DS",
                "ai": "CSE-AIML",
                "ml": "CSE-AIML",
                "ece": "ECE",
                "eee": "EEE",
                "it": "IT",
                "mech": "Mechanical",
                "civil": "Civil",
                "mba": "MBA",
            }
            found_dept = False
            for key, label in deps.items():
                if key in query_lower:
                    search_key = (
                        key
                        if key in ["cse", "ece", "eee", "it", "mech", "civil", "mba"]
                        else "cse-" + key if "cse" not in key else key
                    )
                    hod_name = self.data["personnel"]["hod"].get(search_key)
                    if not hod_name:
                        hod_name = self.data["personnel"]["hod"].get(key)

                    if hod_name:
                        return f"HOD of {label}: {hod_name}"
                        found_dept = True
                        break

            if not found_dept:
                return (
                    "Which department's HOD are you looking for?"
                    " (e.g., CSE, ECE, Mechanical)"
                )

        if any(
            word in query_lower
            for word in ["courses", "branches", "groups", "programmes", "programs"]
        ):
            ug = ", ".join(self.data["courses"]["ug"])
            pg = ", ".join(self.data["courses"]["pg"])
            return (
                "**Courses Offered:**\n\n🎓 **B.Tech:**"
                f" {ug}\n\n🎓 **M.Tech/MBA:** {pg}"
            )

        if any(
            word in query_lower
            for word in [
                "timing",
                "timings",
                "hours",
                "schedule",
                "time table",
                "timetable",
            ]
        ):
            lunch = self.data["timings"].get("lunch_break", "")
            hours = self.data["timings"]["working_hours"]
            return f"**College Timings:**\n\n🕐 {hours}\n\n**Lunch Break:** {lunch}"

        if any(
            word in query_lower
            for word in [
                "address",
                "location",
                "where is",
                "where are",
                "where the college",
                "where is tkrcet",
                "situated",
                "located",
            ]
        ):
            h = self.data["history"]
            return (
                f"**TKRCET Location:**\n\n📍 {h['location']}\n\n"
                f"**Established:** {h['established']}\n**Affiliation:**"
                f" {h['affiliation']}\n**Status:** {h['status']}\n"
                f"**Campus Size:** {h['campus_size']}"
            )

        if "fee" in query_lower and (
            "structure" in query_lower
            or "how much" in query_lower
            or "cost" in query_lower
        ):
            f = self.data["fees"]
            note = f.get(
                "note", "Fees are subject to change as per government regulations."
            )
            return (
                f"**Fee Structure (Approximate):**\n\n• **B.Tech:**"
                f" {f['btech']}\n• **M.Tech:** {f['mtech']}\n• **MBA:**"
                f" {f['mba']}\n\n• **Hostel:** {f['hostel']}\n• **Transport:**"
                f" {f['transport']}\n\n_{note}_"
            )

        # --- Exam / Semester / Supply fee ---
        if any(
            w in query_lower
            for w in [
                "exam fee",
                "semester fee",
                "sem fee",
                "supply fee",
                "examination fee",
            ]
        ):
            ss = self.data.get("student_services", {})
            return ss.get(
                "exam_fee_payment",
                "Please contact the Accounts Department for exam fee payment details.",
            )

        # --- Tuition / Admission fee payment procedure ---
        if "fee" in query_lower and any(
            w in query_lower
            for w in ["pay", "payment", "submit", "how to pay", "how do i pay"]
        ):
            ss = self.data.get("student_services", {})
            return ss.get(
                "fee_payment",
                "Please contact the Accounts Department for fee payment details.",
            )

        # --- Transport fee ---
        if any(
            w in query_lower for w in ["transport fee", "bus fee", "transportation fee"]
        ):
            ss = self.data.get("student_services", {})
            return ss.get(
                "transportation_fee", self.data["facilities"]["transport"]["contact"]
            )

        # --- Hostel fee ---
        if any(w in query_lower for w in ["hostel fee", "accommodation fee"]):
            ss = self.data.get("student_services", {})
            return ss.get(
                "hostel_fee",
                "Please contact the Hostel Office for fee payment details.",
            )

        # --- Bonafide Certificate ---
        if any(
            w in query_lower
            for w in [
                "bonafide",
                "bona fide",
                "bonafide certificate",
                "bonafide letter",
            ]
        ):
            ss = self.data.get("student_services", {})
            return ss.get(
                "bonafide_certificate",
                "Please visit the college office to apply for a bonafide certificate.",
            )

        # --- Scholarship ---
        if any(
            w in query_lower
            for w in ["scholarship", "fellowships", "stipend", "financial aid"]
        ):
            ss = self.data.get("student_services", {})
            return ss.get(
                "scholarship",
                "Please visit the scholarship window in the main block for details.",
            )

        if any(
            word in query_lower
            for word in [
                "library head",
                "head of library",
                "librarian",
                "library in charge",
                "library staff",
            ]
        ):
            lib_head = self.data.get("personnel", {}).get("library_head", None)
            if lib_head:
                return f"**Head of Library (Librarian):** {lib_head}"

        if any(
            word in query_lower
            for word in [
                "placement head",
                "head of placement",
                "placement officer",
                "tpo",
                "t&p",
                "placement cell head",
                "placement in charge",
            ]
        ):
            ph = self.data.get("personnel", {}).get("placement_head", None)
            if ph:
                return f"**Training & Placement Officer:** {ph}"

        if any(word in query_lower for word in ["transport", "bus", "buses", "route"]):
            t = self.data["facilities"]["transport"]
            return (
                f"**College Transport:**\n\n{t['details']}\n\n"
                f"**Routes:** {t['routes']}\n\n{t['contact']}"
            )

        if "canteen" in query_lower or "food" in query_lower:
            c = self.data["facilities"]["canteen"]
            return (
                f"**{c['name']}**\n\n{c['description']}\n\n**Menu:**"
                f" {c['menu']}\n**Timings:** {c['timings']}"
            )
        # Semantic Match
        query_embedding = self.kb_encoder.encode([query], show_progress_bar=False)
        similarities = cosine_similarity(query_embedding, self.kb_embeddings)[0]

        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]

        CONFIDENCE_THRESHOLD = 0.75
        if best_score < CONFIDENCE_THRESHOLD:
            return None

        matched_entry = self.kb_entries[best_idx]
        category = matched_entry["category"]
        key = matched_entry["key"]
        value = matched_entry["value"]

        if category == "personnel":
            if key == "principal":
                return f"Principal: {value}"
            elif key == "vice_principal":
                return f"Vice Principal: {value}"
            elif key == "secretary":
                return f"Secretary: {value}"
            elif key == "chairman":
                return f"Chairman: {value}"
            elif key == "dean_academics":
                return f"Dean Academics: {value}"
            elif "hod" in key:
                dept = key.split(".")[-1] if "." in key else key
                return f"HOD of {dept.upper()}: {value}"
            else:
                return value

        elif category == "timings":
            if key == "working_hours":
                lunch = self.data["timings"].get("lunch_break", "")
                return f"Working Hours: {value}. Lunch Break: {lunch}."
            else:
                return value

        elif category == "history" or key in [
            "location",
            "established",
            "affiliation",
            "status",
            "campus_size",
        ]:
            h = self.data["history"]
            if (
                key == "location"
                or "address" in query.lower()
                or "where" in query.lower()
            ):
                return (
                    f"Location: {h['location']}. Established: "
                    f"{h['established']}. Affiliation: {h['affiliation']}."
                    f" Status: {h['status']}. Campus Size: {h['campus_size']}"
                )
            elif key == "established":
                return f"Established: {value}"
            elif key == "affiliation":
                return f"Affiliation: {value}"
            elif key == "status":
                return f"Status: {value}"
            else:
                return value

        elif category == "transport" or "transport" in key:
            t = self.data["facilities"]["transport"]
            return (
                f"**College Transport:**\n{t['details']}\n\n"
                f"**Routes:** {t['routes']}\n\n{t['contact']}"
            )

        elif category == "canteen" or "canteen" in key:
            c = self.data["facilities"]["canteen"]
            return (
                f"**{c['name']}**\n\n{c['description']}\n\n**Menu:**"
                f" {c['menu']}\n**Timings:** {c['timings']}"
            )

        elif category == "campus_life" or key == "events" or key == "clubs":
            cl = self.data["activities"]["campus_life"]
            return (
                f"**Campus Life at TKRCET**\n\n{cl['overview']}\n\n"
                f"**Events:** {cl['events']}\n**Clubs:** "
                f"{cl['clubs']}\n\n{cl['environment']}"
            )

        elif category == "ncc":
            ncc = self.data["activities"]["ncc"]
            return (
                f"**{ncc['name']}**\n\n{ncc['description']}\n\n"
                f"**Benefits:** {ncc['benefits']}"
            )

        elif category == "nss":
            nss = self.data["activities"]["nss"]
            return (
                f"**{nss['name']}**\n\n{nss['description']}\n\n"
                f"**Motto:** \"{nss['motto']}\""
            )

        # Special handling for society (Chairman/Secretary)
        if category == "society":
            if "chairman" in key.lower():
                return f"The Chairman of TKRCET is {self.data['society']['chairman']}."
            if "secretary" in key.lower():
                return (
                    f"The Secretary of TKRCET is {self.data['society']['secretary']}."
                )
            # Fallback for general society info
            return (
                "TKRCET is managed by the Teegala Krishna Reddy "
                f"Educational Society. Key members include Chairman "
                f"{self.data['society']['chairman']}"
                f" and Secretary {self.data['society']['secretary']}."
            )

        elif category == "courses":
            ug = ", ".join(self.data["courses"]["ug"])
            pg = ", ".join(self.data["courses"]["pg"])
            return f"UG Programs: {ug}. PG Programs: {pg}."

        elif category == "fees" or "fee" in key:
            f = self.data["fees"]
            note = f.get(
                "note", "Fees are subject to change as per government regulations."
            )
            return (
                f"**Fee Structure (Approximate):**\n\n• **B.Tech:**"
                f"{f['btech']}\n• **M.Tech:** {f['mtech']}\n• **MBA:** "
                f"{f['mba']}\n\n• **Hostel:** {f['hostel']}\n• "
                f"**Transport:** {f['transport']}\n\n_{note}_"
            )

        elif category == "exam_info":
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

        if "statistics" in self.data:
            stats = self.data["statistics"]
            lines.append("\nFAST FACTS:")
            lines.append(f"• Total Students: {stats['total_students']}")
            lines.append(
                f"• Placed Students: {stats['placed_students']}"
                f" (Rate: {stats['placement_rate']})"
            )
            lines.append(f"• Top Recruiters: {stats['top_recruiters']}")

        return "\n".join(lines)
