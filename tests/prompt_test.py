# flake8: noqa: E501

import json
import os
import re
import time

import pandas as pd
import requests
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font
from sentence_transformers import SentenceTransformer, util

API_URL = "http://127.0.0.1:8000/query"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
JUDGE_MODEL = "gemma3:1b"

print("Loading similarity model (all-MiniLM-L6-v2) for evaluation...")
sim_model = SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------------------------------------
# PROMPTS
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
]


# --------------------------------------------------------------
# LLM JUDGE
# --------------------------------------------------------------
def llm_judge(query, answer, context):

    if not answer or answer.startswith("ERROR") or answer.startswith("CONNECTION"):
        return 0.0, 0.0, 0.0

    judge_prompt = f"""
You are a strict quality evaluator for a college chatbot.

Question: {query}

Context:
{context[:600] if context else "No context"}

Answer:
{answer[:600]}

Return ONLY JSON:

{{"faithfulness":0-1,"relevance":0-1,"completeness":0-1}}
"""

    try:

        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": JUDGE_MODEL,
                "prompt": judge_prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 80},
            },
            timeout=30,
        )

        if resp.status_code == 200:

            raw = resp.json().get("response", "").strip()
            match = re.search(r"\{.*?\}", raw, re.DOTALL)

            if match:

                scores = json.loads(match.group())

                f = float(scores.get("faithfulness", 0.5))
                r = float(scores.get("relevance", 0.5))
                c = float(scores.get("completeness", 0.5))

                return f, r, c

    except Exception as e:
        print("Judge error:", e)

    return 0.5, 0.5, 0.5


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

    except Exception:

        print("⏳ Waiting for backend...")
        time.sleep(1)


# --------------------------------------------------------------
# LOG FILE
# --------------------------------------------------------------

LOG_FILE = "logs/response_log.xlsx"
EVAL_SHEET = "Evaluation"

os.makedirs("logs", exist_ok=True)

EVAL_COLS = [
    "Timestamp",
    "Prompt",
    "Bot Answer",
    "Source",
    "Retrieval Confidence (%)",
    "Latency (s)",
    "Server Time (s)",
    "Faithfulness %",
    "Relevance %",
    "Completeness %",
    "Accuracy %",
]


def open_sheet():

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


def append_row(row):

    try:

        wb, ws = open_sheet()

        ws.append(row)

        wb.save(LOG_FILE)

    except Exception as e:

        print("Save error:", e)


# --------------------------------------------------------------
# RUN TESTS
# --------------------------------------------------------------

results = []

for prompt in test_prompts:

    print("\nTesting:", prompt)

    start = time.time()

    try:

        resp = requests.post(API_URL, json={"message": prompt}, timeout=60)

        latency = round(time.time() - start, 3)

        if resp.status_code == 200:

            data = resp.json()

            answer = data.get("answer", "")

            source = data.get("source", "Unknown")

        else:

            answer = "ERROR"
            source = "Unknown"

    except Exception as e:

        latency = round(time.time() - start, 3)

        answer = "CONNECTION ERROR " + str(e)

        source = "Unknown"

    print("Running LLM judge...")

    faith, rel, comp = llm_judge(prompt, answer, "")

    accuracy = round((faith + rel + comp) / 3 * 100, 2)

    row = [
        time.strftime("%Y-%m-%d %H:%M:%S"),
        prompt,
        answer,
        source,
        "N/A",
        latency,
        "N/A",
        round(faith * 100, 1),
        round(rel * 100, 1),
        round(comp * 100, 1),
        accuracy,
    ]

    append_row(row)

    results.append(accuracy)

    print("Accuracy:", accuracy)


# --------------------------------------------------------------
# SUMMARY
# --------------------------------------------------------------

if results:

    avg_acc = sum(results) / len(results)

    print(
        "\nAll results saved to",
        EVAL_SHEET,
        "sheet in",
        LOG_FILE,
    )

    print("\nAverage Accuracy:", round(avg_acc, 2))

else:

    print("No results.")
