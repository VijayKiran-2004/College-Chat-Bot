
import pandas as pd
import os

log_path = 'logs/response_log.xlsx'
df = pd.read_excel(log_path)
print(f"Columns: {list(df.columns)}\n")

recent = df.tail(12)
for i, row in recent.iterrows():
    print(f"=== Entry {i+1} ===")
    for col in df.columns:
        val = str(row[col])
        if col == 'Bot Response':
            val = val[:150] + "..."
        print(f"  {col}: {val}")
    print()
