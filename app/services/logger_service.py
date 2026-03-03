import os
from datetime import datetime
from pathlib import Path
from openpyxl import Workbook, load_workbook

class ResponseLogger:
    def __init__(self, log_dir=None, filename="response_log.xlsx"):
        """
        Initialize the ResponseLogger.
        
        Args:
            log_dir (str): Directory where the log file will be stored.
            filename (str): Name of the Excel file.
        """
        if log_dir is None:
            # Default to logs directory in project root
            self.log_dir = Path(__file__).resolve().parent.parent.parent / "logs"
        else:
            self.log_dir = Path(log_dir)
            
        self.log_file = self.log_dir / filename
        self._ensure_log_file()

    def _ensure_log_file(self):
        """Ensure the log directory and file exist with proper headers."""
        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
        columns = ["Timestamp", "User Query", "Bot Response", "Time Taken (s)", "Session ID", "Source", 
                   "Retrieval Confidence (%)", "Faithfulness", "Answer Relevance", 
                   "Cross-Validation (SQL vs. RAG)", "Link Validity", "Accuracy"]
            
        if not self.log_file.exists():
            wb = Workbook()
            ws = wb.active
            ws.append(columns)
            wb.save(self.log_file)
        else:
            # Check if columns need updating
            try:
                wb = load_workbook(self.log_file)
                ws = wb.active
                headers = [cell.value for cell in ws[1]]
                
                if "Faithfulness" not in headers:
                    # Append new headers
                    for col in columns:
                        if col not in headers:
                            ws.cell(row=1, column=len(headers)+1, value=col)
                            headers.append(col)
                    wb.save(self.log_file)
            except Exception as e:
                print(f"Error checking log file headers: {e}")

    def log_response(self, user_query, bot_response, time_taken, session_id="N/A", 
                     source="N/A", confidence="N/A", faithfulness=1.0, relevance=1.0, 
                     cross_val="OK", link_val="N/A", accuracy=1.0):
        """
        Log a query and response pair to the Excel file with 4-Pillar metrics.
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Load existing workbook
            if self.log_file.exists():
                wb = load_workbook(self.log_file)
                ws = wb.active
            else:
                self._ensure_log_file()
                wb = load_workbook(self.log_file)
                ws = wb.active
            
            # Append new row
            ws.append([
                timestamp, user_query, bot_response, f"{time_taken:.4f}", session_id, source,
                confidence, faithfulness, relevance, cross_val, link_val, accuracy
            ])
            
            wb.save(self.log_file)
            print(f"✓ Logged response to: {self.log_file}")
        except PermissionError:
            print(f"⚠ ERROR: Could not write to {self.log_file}. Is the file open in Excel?")
        except Exception as e:
            print(f"Error logging response: {e}")
