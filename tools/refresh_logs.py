"""
Refresh Logs - Archives the current log and creates a fresh multi-sheet Excel file.
Usage: python tools/refresh_logs.py
"""
import sys, os, shutil
from datetime import datetime

if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font

log_dir   = 'logs'
log_path  = f'{log_dir}/response_log.xlsx'
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(log_dir, exist_ok=True)

# --- Archive the existing file ---
if os.path.exists(log_path):
    counts = {}
    try:
        wb_old = load_workbook(log_path)
        for sheet in wb_old.sheetnames:
            ws = wb_old[sheet]
            counts[sheet] = ws.max_row - 1  # minus header row
    except Exception:
        counts = {"(unreadable)": "?"}

    archive_path = f"{log_dir}/response_log_{timestamp}.xlsx"
    shutil.move(log_path, archive_path)
    summary = ", ".join([f"{s}: {n} entries" for s, n in counts.items()])
    print(f"  Archived -> {archive_path}")
    print(f"  ({summary})")
else:
    print("  No existing log found. Creating fresh one.")

# --- Create a fresh workbook with both sheets ---
wb = Workbook()

# Sheet 1: Production (backend.py)
ws_prod = wb.active
ws_prod.title = "Production"
prod_cols = [
    "Timestamp", "User Query", "Bot Response",
    "Time Taken (s)", "Session ID", "Source"
]
ws_prod.append(prod_cols)
for cell in ws_prod[1]:
    cell.font = Font(bold=True)

# Sheet 2: Evaluation (prompt_test.py)
ws_eval = wb.create_sheet("Evaluation")
eval_cols = [
    "Timestamp", "Prompt", "Bot Answer", "Source",
    "Retrieval Confidence (%)", "Latency (s)", "Server Time (s)",
    "Faithfulness % (LLM)", "Relevance % (LLM)", "Completeness % (LLM)",
    "BERTScore F1 %", "Link Validity", "Accuracy %"
]
ws_eval.append(eval_cols)
for cell in ws_eval[1]:
    cell.font = Font(bold=True)

wb.save(log_path)
print(f"\n  Fresh log created: {log_path}")
print(f"  Sheets: 'Production' (backend logs) | 'Evaluation' (test results)")
print(f"  Ready for new queries and evaluations!")
