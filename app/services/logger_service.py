import os
from datetime import datetime
from pathlib import Path
from openpyxl import Workbook, load_workbook

class ResponseLogger:
    def __init__(self, log_dir="logs", filename="response_log.xlsx"):
        """
        Initialize the ResponseLogger.
        
        Args:
            log_dir (str): Directory where the log file will be stored.
            filename (str): Name of the Excel file.
        """
        self.log_dir = Path(log_dir)
        self.log_file = self.log_dir / filename
        self._ensure_log_file()

    def _ensure_log_file(self):
        """Ensure the log directory and file exist with proper headers."""
        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
        if not self.log_file.exists():
            wb = Workbook()
            ws = wb.active
            ws.append(["Timestamp", "User Query", "Bot Response", "Time Taken (s)", "Session ID"])
            wb.save(self.log_file)

    def log_response(self, user_query, bot_response, time_taken, session_id="N/A"):
        """
        Log a query and response pair to the Excel file.
        
        Args:
            user_query (str): The user's input message.
            bot_response (str): The bot's generated response.
            time_taken (float): Time taken to generate the response in seconds.
            session_id (str): The session identifier.
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Load existing workbook
            if self.log_file.exists():
                wb = load_workbook(self.log_file)
                ws = wb.active
            else:
                wb = Workbook()
                ws = wb.active
                ws.append(["Timestamp", "User Query", "Bot Response", "Time Taken (s)", "Session ID"])
            
            # Append new row
            ws.append([timestamp, user_query, bot_response, f"{time_taken:.4f}", session_id])
            
            wb.save(self.log_file)
        except Exception as e:
            print(f"Error logging response: {e}")
