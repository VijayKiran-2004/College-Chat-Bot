import requests
import pandas as pd
import time
import json
import os
import re
from sentence_transformers import SentenceTransformer, util

API_URL = "http://127.0.0.1:8000/query"

# Initialize Similarity Model (Cache it to avoid reloading)
print("Loading similarity model (all-MiniLM-L6-v2) for evaluation...")
sim_model = SentenceTransformer('all-MiniLM-L6-v2')

# --------------------------------------------------------------
# ALL YOUR PROMPTS (unchanged, exactly as you wrote them)
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
    "do college provide traiing before drive?",
    "who is head of placement cell?",
    "what is the highest package offered in 2024?",
    "who got highest package in 2025?",
    "what is full name of college?",
    "Do college have any sports facilities?",
    "How to take admission in college hostel?",
    "give me contact details of college hostel",
    "in which rout college bus is available?",
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
    "whcih company offered maximum packages in 2024?",
    "what is the average package offered in 2024?",
    "what percentage of female students got placed in 2025?",
    "how many students placed in savantis?",
    "how many CSD students got placed in kasmo in 2025?",
    "what is the minimum CGPA required for placements?",
    "do college allow students to sit for placements if they have backlogs?",
    "how many students got internships in 2024?",
    "do college have any international collaborations?",
    "do college give attendence relaxation for sports events?",
    "who is the head of computer science department?",
    "do college have any cultural clubs?",
    "any events organized by college in last year?",
    "what is CGPA of student with roll number 21k91a0572?",
    "do college organize any technical fests?"
]

def check_link_validity(text):
    """Simple check to find markdown links and verify they are reachable"""
    links = re.findall(r'\[.*?\]\((https?://.*?)\)', text)
    if not links:
        return "N/A"
    
    valid_count = 0
    for link in links:
        try:
            # Quick HEAD request to check URL
            r = requests.head(link, timeout=5, allow_redirects=True)
            if r.status_code < 400:
                valid_count += 1
        except:
            continue
    
    return f"{valid_count}/{len(links)} Valid"

def calculate_similarity(text1, text2):
    """Utility to calculate cosine similarity between two strings"""
    if not text1 or not text2:
        return 0.0
    emb1 = sim_model.encode(text1, convert_to_tensor=True)
    emb2 = sim_model.encode(text2, convert_to_tensor=True)
    return round(float(util.pytorch_cos_sim(emb1, emb2)), 4)

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

# --------------------------------------------------------------
# RUN TESTS
# --------------------------------------------------------------
results = []

for prompt in test_prompts:
    print(f"\n→ Testing: {prompt}")

    start = time.time()
    retrieval_confidence = "N/A"
    retrieved_context = []
    source = "Unknown"
    
    try:
        # Request with stream=True to handle possible StreamingResponse (SSE)
        resp = requests.post(API_URL, json={"message": prompt, "session_id": None}, timeout=60, stream=True)
        latency = round(time.time() - start, 3)

        if resp.status_code == 200:
            content_type = resp.headers.get("Content-Type", "")
            
            if "text/event-stream" in content_type:
                # Handle Streaming Response (SSE)
                full_answer = ""
                for line in resp.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith("data: "):
                            try:
                                data = json.loads(line_str[6:])
                                if data.get("type") == "metadata":
                                    retrieval_confidence = data.get("confidence", "N/A")
                                    retrieved_context = data.get("context", [])
                                    source = data.get("source", "RAG")
                                if "chunk" in data:
                                    full_answer += data["chunk"]
                                if data.get("done"):
                                    break
                            except:
                                continue
                answer = full_answer
            else:
                # Handle standard JSON response
                try:
                    json_resp = resp.json()
                    answer = json_resp.get("answer", "")
                    source = json_resp.get("source", "SQL/KB")
                    retrieval_confidence = json_resp.get("confidence", "N/A")
                except:
                    answer = f"ERROR: Could not parse JSON. Body: {resp.text[:100]}"
        else:
            answer = f"ERROR ({resp.status_code})"
    except Exception as e:
        latency = round(time.time() - start, 3)
        answer = f"CONNECTION ERROR: {str(e)}"

    # 1. Faithfulness (Similarity between answer and context)
    context_str = " ".join(retrieved_context)
    if retrieved_context:
        faithfulness = calculate_similarity(answer, context_str)
    else:
        # If SQL/KB, faithfulness is 1.0. If RAG but no context, it's 0.0 (error in retrieval)
        faithfulness = 1.0 if source in ["SQL Database", "SQL/KB", "Instant Response"] else 0.0

    # 2. Answer Relevance (Similarity between prompt and answer)
    relevance = calculate_similarity(prompt, answer) if not answer.startswith("ERROR") else 0.0

    # 3. Cross-Validation (SQL vs RAG Consistency)
    cross_val_score = 1.0
    cross_val_label = "OK"
    if source == "RAG" and any(kw in prompt.lower() for kw in ["list", "how many", "statistics", "placed in"]):
        cross_val_label = "Review (Suggest SQL)"
        cross_val_score = 0.5
    elif source == "SQL Database" and any(kw in prompt.lower() for kw in ["about", "procedure", "how to"]):
        cross_val_label = "Review (Suggest RAG)"
        cross_val_score = 0.5

    # 4. Link Validity
    link_results = check_link_validity(answer)
    if link_results == "N/A":
        link_score = 1.0
    else:
        match = re.search(r'(\d+)/(\d+)', link_results)
        link_score = int(match.group(1)) / int(match.group(2)) if match else 0.0

    # Calculate Accuracy (Weighted Average: Relevance 40%, Faithfulness 30%, others 15%)
    # This makes the score more "honest" about factual relevance
    accuracy = round((faithfulness * 0.3) + (relevance * 0.4) + (cross_val_score * 0.15) + (link_score * 0.15), 4)

    print(f"✓ Source: {source} | Latency: {latency}s")
    print(f"  [Metrics] Faithfulness: {faithfulness} | Relevance: {relevance} | Link: {link_results}")
    print(f"  [Final] Accuracy Score: {accuracy}")

    results.append({
        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "Prompt": prompt,
        "Bot Answer": answer,
        "Source": source,
        "Retrieval Confidence (%)": retrieval_confidence,
        "Faithfulness": faithfulness,
        "Answer Relevance": relevance,
        "Cross-Validation (SQL vs. RAG)": cross_val_label,
        "Link Validity": link_results,
        "Accuracy": accuracy,
        "Latency (s)": latency
    })
    
    # Small delay between tests to let backend breathe
    time.sleep(0.5)

# --------------------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------------------
df = pd.DataFrame(results)
log_path = "logs/response_log.xlsx"
os.makedirs("logs", exist_ok=True)
try:
    df.to_excel(log_path, index=False)
    print(f"\n✓ Results saved to {log_path} ({len(results)} entries)")
except Exception as e:
    print(f"\n❌ Failed to save to Excel: {e}")
    df.to_csv("logs/response_log.csv", index=False)
    print(f"✓ Results saved to logs/response_log.csv instead ({len(results)} entries)")
