"""
Script to refresh/clear the response log Excel file
This will backup the existing log and create a fresh one with headers only
"""
import os
from pathlib import Path
from datetime import datetime
from openpyxl import Workbook

def refresh_logs():
    log_dir = Path("logs")
    log_file = log_dir / "response_log.xlsx"
    
    # Create backup if log file exists
    if log_file.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = log_dir / f"response_log_backup_{timestamp}.xlsx"
        
        # Copy to backup
        import shutil
        shutil.copy2(log_file, backup_file)
        print(f"✓ Backup created: {backup_file}")
        
        # Delete original
        os.remove(log_file)
        print(f"✓ Deleted old log file")
    
    # Create fresh log file with headers
    wb = Workbook()
    ws = wb.active
    
    # Add headers
    headers = ["Timestamp", "User Query", "Bot Response", "Time Taken (s)", "Session ID", "Source", "Accuracy"]
    ws.append(headers)
    
    # Save
    wb.save(log_file)
    print(f"✓ Created fresh log file: {log_file}")
    print("\nLog file has been refreshed successfully!")

if __name__ == "__main__":
    print("=" * 70)
    print("REFRESHING RESPONSE LOG FILE")
    print("=" * 70)
    print()
    
    refresh_logs()
    
    print()
    print("=" * 70)
