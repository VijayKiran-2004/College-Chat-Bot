"""
Refresh Logs - Clears the old response log and creates a fresh Excel sheet.
Usage: python refresh_logs.py
"""
import sys, os
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from openpyxl import Workbook

log_path = 'logs/response_log.xlsx'

# Show old stats before clearing
if os.path.exists(log_path):
    import pandas as pd
    df = pd.read_excel(log_path)
    print(f"  Old log had {len(df)} entries. Clearing...")
else:
    print("  No existing log found.")

# Create fresh workbook with headers
os.makedirs('logs', exist_ok=True)
wb = Workbook()
ws = wb.active
ws.title = "Response Log"

headers = ["Timestamp", "User Query", "Bot Response", "Time Taken (s)", "Session ID", "Source", 
           "Retrieval Confidence (%)", "Faithfulness", "Answer Relevance", 
           "Cross-Validation (SQL vs. RAG)", "Link Validity", "Accuracy"]
ws.append(headers)

# Style header row
from openpyxl.styles import Font
for cell in ws[1]:
    cell.font = Font(bold=True)

wb.save(log_path)
print(f"  ✓ Fresh log created at: {log_path}")
print(f"  Ready for new queries!")
