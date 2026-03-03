import requests
import pandas as pd
import time
import json
import os
import re
from sentence_transformers import SentenceTransformer, util

API_URL    = "http://127.0.0.1:8000/query"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
JUDGE_MODEL = "gemma3:1b"  # Small local model used only for evaluation

# Initialize Similarity Model (for BERTScore-style eval)
print("Loading similarity model (all-MiniLM-L6-v2) for evaluation...")
sim_model = SentenceTransformer('all-MiniLM-L6-v2')

# --------------------------------------------------------------
# ALL YOUR PROMPTS
# --------------------------------------------------------------
test_prompts = [
    "What is the attendance criteria?",
    "What is the NAAC grade of the college?",
    "Tell me placement statistics.",
    "List all CSE students.",
    "where the college is located?",
    "How many students got placed in 2024?",
    "What courses are offered in the College?",
    "How to check the results?",
    "what is the minimum attendance required?",
    "Do college have any dress code or uniform",
    "who got placed in savantis in 2024",
    "What is the fee structure for MBA program?",
    "Give me the list of students placed in 2023",
    "who is topper of CSE department in 2024",
    "how many cse students got placed in jbm group?",
    "what is the female pass percentage in 2025?",
    "who is head of library?",
    "which companies visited for placements in 2024?",
    "do college have hostel facilities?",
    "tell me about the college infrastructure.",
    "Do college have transport facilities?",
    "what is the procedure to apply for scholarships?",
    "do college provide training before drive?",
    "who is head of placement cell?",
    "what is the highest package offered in 2024?",
    "who got highest package in 2025?",
    "what is full name of college?",
    "Do college have any sports facilities?",
    "How to take admission in college hostel?",
    "give me contact details of college hostel",
    "in which route college bus is available?",
    "what is sem fee for computer science engineering?",
    "who is the principal of the college?",
    "how to apply for supply exams?",
    "what is the library timing?",
    "how many books are there in college library?",
    "do college have any research publications?",
    "do college got any government projects sanctioned?",
    "do college have an entrepreneurship cell?",
    "do college provide internships to students?",
    "what is fee structure for btech program?",
    "how many departments are there in college?",
    "do sem exams have revaluation facility?",
    "what is the procedure to apply for revaluation?",
    "tell me about college accreditation status?",
    "is college affiliated to any university?",
    "who is dean of Research and development?",
    "Do college have wifi and lab facilities?",
    "what is the procedure to apply for internships?",
    "which company offered maximum packages in 2024?",
    "what is the average package offered in 2024?",
    "what percentage of female students got placed in 2025?",
    "how many students placed in savantis?",
    "how many CSD students got placed in kasmo in 2025?",
    "what is the minimum CGPA required for placements?",
    "do college allow students to sit for placements if they have backlogs?",
    "how many students got internships in 2024?",
    "do college have any international collaborations?",
    "do college give attendance relaxation for sports events?",
    "who is the head of computer science department?",
    "do college have any cultural clubs?",
    "any events organized by college in last year?",
    "what is CGPA of student with roll number 21k91a0572?",
    "do college organize any technical fests?"
]

# --------------------------------------------------------------
# LLM-AS-JUDGE: Ask Gemma 1B to score the response
# --------------------------------------------------------------
def llm_judge(query, answer, context):
    """Ask a local small model to judge the response quality. Returns faithfulness, relevance, completeness (0.0-1.0)."""
    if not answer or answer.startswith("ERROR") or answer.startswith("CONNECTION"):
        return 0.0, 0.0, 0.0

    judge_prompt = f"""You are a strict quality evaluator for a college chatbot.

Question: {query}
Context (what the bot knew): {context[:600] if context else "No context (SQL/KB answer)"}
Bot's Answer: {answer[:600]}

Rate the answer ONLY based on the above. Be very strict. Output ONLY valid JSON, nothing else:
{{"faithfulness": <0.0-1.0>, "relevance": <0.0-1.0>, "completeness": <0.0-1.0>}}

- faithfulness: Is every claim in the answer supported by the context?
- relevance: Does the answer directly address the question?
- completeness: Does the answer cover all aspects of the question?"""

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": JUDGE_MODEL, "prompt": judge_prompt, "stream": False,
                  "options": {"temperature": 0.0, "num_predict": 80}},
            timeout=30
        )
        if resp.status_code == 200:
            raw = resp.json().get("response", "").strip()
            # Extract JSON from LLM output robustly
            match = re.search(r'\{.*?\}', raw, re.DOTALL)
            if match:
                scores = json.loads(match.group())
                f = min(1.0, max(0.0, float(scores.get("faithfulness", 0.5))))
                r = min(1.0, max(0.0, float(scores.get("relevance", 0.5))))
                c = min(1.0, max(0.0, float(scores.get("completeness", 0.5))))
                return f, r, c
    except Exception as e:
        print(f"    ⚠ Judge error: {e}")
    return 0.5, 0.5, 0.5  # Neutral fallback

# --------------------------------------------------------------
# BERTSCORE: Token-level semantic overlap (answer vs context)
# --------------------------------------------------------------
def bertscore_f1(answer, context):
    """Compute a BERTScore-style F1 using sentence-transformers. No extra models needed."""
    if not answer or not context:
        return 1.0  # SQL/KB with no context defaults to 1.0 (graded by judge instead)

    # Split into sentences for more granular matching
    ans_sents = [s.strip() for s in re.split(r'[.!?]', answer) if len(s.strip()) > 10]
    ctx_sents = [s.strip() for s in re.split(r'[.!?]', context) if len(s.strip()) > 10]

    if not ans_sents or not ctx_sents:
        return float(util.pytorch_cos_sim(
            sim_model.encode(answer, convert_to_tensor=True),
            sim_model.encode(context, convert_to_tensor=True)
        ).item())

    ans_embs = sim_model.encode(ans_sents, convert_to_tensor=True)
    ctx_embs = sim_model.encode(ctx_sents, convert_to_tensor=True)
    sim_matrix = util.pytorch_cos_sim(ans_embs, ctx_embs)

    # Precision: for each answer sentence, find best matching context sentence
    precision = float(sim_matrix.max(dim=1).values.mean())
    # Recall: for each context sentence, find best matching answer sentence
    recall    = float(sim_matrix.max(dim=0).values.mean())

    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return round(f1, 4)

# --------------------------------------------------------------
# LINK VALIDITY
# --------------------------------------------------------------
def check_link_validity(text):
    links = re.findall(r'\[.*?\]\((https?://.*?)\)', text)
    if not links:
        return "N/A", 1.0
    valid_count = 0
    for link in links:
        try:
            r = requests.head(link, timeout=5, allow_redirects=True)
            if r.status_code < 400:
                valid_count += 1
        except:
            continue
    score = valid_count / len(links)
    return f"{valid_count}/{len(links)} Valid", round(score, 4)

# --------------------------------------------------------------
# SOURCE APPROPRIATENESS
# --------------------------------------------------------------
def source_appropriateness(query, source):
    """Rule-based check: did the right system handle this query?"""
    q = query.lower()
    is_count_query = any(kw in q for kw in ["how many", "list all", "who placed", "placed in", "give me list"])
    is_fact_query  = any(kw in q for kw in ["what is", "who is", "full name", "grade", "fee structure", "timing"])
    is_proc_query  = any(kw in q for kw in ["how to", "procedure", "apply for", "steps", "process"])

    if is_count_query and source == "SQL Database":
        return 1.0
    if is_fact_query and source in ["Knowledge Base", "RAG"]:
        return 1.0
    if is_proc_query and source == "RAG":
        return 1.0
    if source == "Unknown":
        return 0.0
    return 0.75  # Neutral — can't fully determine

# --------------------------------------------------------------
# WAIT FOR BACKEND
# --------------------------------------------------------------
print("Checking backend status...")
while True:
    try:
        r = requests.get("http://127.0.0.1:8000/health", timeout=3)
        if r.status_code == 200:
            print("✓ Backend is online.\n")
            break
    except:
        print("⏳ Waiting for backend...")
        time.sleep(1)

# Both production and evaluation logs go to the SAME file, different sheets
LOG_FILE  = "logs/response_log.xlsx"
EVAL_SHEET = "Evaluation"
os.makedirs("logs", exist_ok=True)

# -- Imports for incremental save --
from openpyxl import load_workbook, Workbook
from openpyxl.styles import Font

EVAL_COLS = [
    "Timestamp", "Prompt", "Bot Answer", "Source",
    "Retrieval Confidence (%)", "Latency (s)", "Server Time (s)",
    "Faithfulness % (LLM)", "Relevance % (LLM)", "Completeness % (LLM)",
    "BERTScore F1 %", "Link Validity", "Accuracy %"
]

def _open_eval_sheet():
    """Load (or create) the workbook and return (wb, ws) for the Evaluation sheet."""
    if os.path.exists(LOG_FILE):
        wb = load_workbook(LOG_FILE)
    else:
        wb = Workbook()
    if EVAL_SHEET in wb.sheetnames:
        ws = wb[EVAL_SHEET]
    else:
        ws = wb.create_sheet(EVAL_SHEET)
        ws.append(EVAL_COLS)
        for cell in ws[1]:
            cell.font = Font(bold=True)
    return wb, ws

def _append_and_save(row: dict):
    """Append one result row to the Evaluation sheet and save immediately."""
    try:
        wb, ws = _open_eval_sheet()
        ws.append([
            row["Timestamp"], row["Prompt"], row["Bot Answer"], row["Source"],
            row["Retrieval Confidence (%)"], row["Latency (s)"], row["Server Time (s)"],
            row["Faithfulness % (LLM)"], row["Relevance % (LLM)"], row["Completeness % (LLM)"],
            row["BERTScore F1 %"], row["Link Validity"], row["Accuracy %"],
        ])
        wb.save(LOG_FILE)
        print(f"  ✓ Saved to Evaluation sheet ({ws.max_row - 1} rows so far)")
    except PermissionError:
        print(f"  ⚠ Cannot write — is response_log.xlsx open in Excel?")
    except Exception as e:
        print(f"  ⚠ Save error: {e}")

# --------------------------------------------------------------
# RUN TESTS
# --------------------------------------------------------------
results = []

for prompt in test_prompts:
    print(f"\n→ Testing: {prompt}")

    start = time.time()
    retrieval_confidence = "N/A"
    retrieved_context    = []
    source               = "Unknown"
    server_time_taken    = "N/A"  # will be filled from backend response

    try:
        resp = requests.post(API_URL, json={"message": prompt, "session_id": None}, timeout=60, stream=True)
        latency = round(time.time() - start, 3)

        if resp.status_code == 200:
            content_type = resp.headers.get("Content-Type", "")

            if "text/event-stream" in content_type:
                full_answer = ""
                for line in resp.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith("data: "):
                            try:
                                data = json.loads(line_str[6:])
                                if data.get("type") == "metadata":
                                    retrieval_confidence = data.get("confidence", "N/A")
                                    retrieved_context    = data.get("context", [])
                                    source               = data.get("source", "RAG")
                                if "chunk" in data:
                                    full_answer += data["chunk"]
                                if data.get("done"):
                                    server_time_taken = data.get("time_taken", "N/A")
                                    break
                            except:
                                continue
                answer = full_answer
            else:
                try:
                    json_resp = resp.json()
                    answer    = json_resp.get("answer", "")
                    source    = json_resp.get("source", "SQL/KB")
                    retrieval_confidence = json_resp.get("confidence", "N/A")
                    server_time_taken    = json_resp.get("time_taken", "N/A")
                except:
                    answer = f"ERROR: Could not parse JSON."
        else:
            answer = f"ERROR ({resp.status_code})"
    except Exception as e:
        latency = round(time.time() - start, 3)
        answer  = f"CONNECTION ERROR: {str(e)}"

    context_str = " ".join(retrieved_context)

    # --- Metric 1, 2, 3: LLM-as-Judge ---
    print(f"    ⚖ Running LLM judge...")
    faithfulness_llm, relevance_llm, completeness_llm = llm_judge(prompt, answer, context_str)

    # --- Metric 4: BERTScore F1 ---
    bert_f1 = bertscore_f1(answer, context_str)

    # --- Metric 5: Link Validity ---
    link_label, link_score = check_link_validity(answer)

    # --- Final Accuracy % (balanced, no single dominant metric) ---
    src_score = source_appropriateness(prompt, source)
    accuracy_pct = round(
        (faithfulness_llm * 0.30 +
         relevance_llm    * 0.25 +
         completeness_llm * 0.20 +
         bert_f1          * 0.15 +
         src_score        * 0.10) * 100,
        2
    )

    # Convert all 0-1 scores to 0-100%
    faith_pct    = round(faithfulness_llm * 100, 1)
    rel_pct      = round(relevance_llm    * 100, 1)
    complete_pct = round(completeness_llm * 100, 1)
    bert_pct     = round(bert_f1          * 100, 1)

    print(f"✓ Source: {source} | Latency: {latency}s | Server Time: {server_time_taken}s")
    print(f"  [LLM Judge] Faith: {faith_pct}% | Rel: {rel_pct}% | Complete: {complete_pct}%")
    print(f"  [BERTScore] F1: {bert_pct}% | [Links] {link_label}")
    print(f"  [Final Accuracy] {accuracy_pct}%")

    row = {
        "Timestamp":               time.strftime("%Y-%m-%d %H:%M:%S"),
        "Prompt":                  prompt,
        "Bot Answer":              answer,
        "Source":                  source,
        "Retrieval Confidence (%)": retrieval_confidence,
        "Latency (s)":             latency,
        "Server Time (s)":         server_time_taken,
        "Faithfulness % (LLM)":    faith_pct,
        "Relevance % (LLM)":       rel_pct,
        "Completeness % (LLM)":    complete_pct,
        "BERTScore F1 %":          bert_pct,
        "Link Validity":           link_label,
        "Accuracy %":              accuracy_pct,
    }
    results.append(row)
    _append_and_save(row)   # ← save immediately after each prompt

    time.sleep(0.5)

# --------------------------------------------------------------
# FINAL SUMMARY (results already saved row-by-row above)
# --------------------------------------------------------------
if results:
    df = pd.DataFrame(results)
    avg_acc     = df["Accuracy %"].mean()
    avg_latency = df["Latency (s)"].mean()
    print(f"\n✓ All {len(results)} results saved to '{EVAL_SHEET}' sheet in {LOG_FILE}")
    print(f"\n📊 Summary — Avg Accuracy: {avg_acc:.1f}% | Avg Latency: {avg_latency:.2f}s")
else:
    print("\n⚠ No results to summarise.")
