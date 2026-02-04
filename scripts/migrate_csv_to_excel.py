import pandas as pd
import os

csv_file = r"c:\college-buddy - Copy\logs\response_log_cleaned.csv"
excel_file = r"c:\college-buddy - Copy\logs\response_log.xlsx"

try:
    if os.path.exists(csv_file):
        print(f"Reading {csv_file}...")
        df = pd.read_csv(csv_file)
        
        print(f"Converting to {excel_file}...")
        df.to_excel(excel_file, index=False)
        
        print("Conversion successful.")
    else:
        print(f"Source file {csv_file} not found.")

except Exception as e:
    print(f"An error occurred during migration: {e}")
