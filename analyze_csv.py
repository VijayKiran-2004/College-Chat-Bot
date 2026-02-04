import csv
import sys

filename = r"c:\college-buddy - Copy\logs\response_log_cleaned.csv"

try:
    with open(filename, 'r', encoding='utf-8') as f:
        # Use csv.reader to handle quoted multiline fields correctly
        reader = csv.reader(f)
        try:
            header = next(reader)
            print(f"Header: {header}")
            expected_cols = len(header)
        except StopIteration:
            print("File is empty")
            sys.exit(0)

        row_count = 0
        mismatched_rows = 0
        repeated_headers = 0
        
        for i, row in enumerate(reader, start=2): # Start at 2 since header was 1
            row_count += 1
            if len(row) != expected_cols:
                print(f"Row {i} has {len(row)} columns, expected {expected_cols}. Content: {row}")
                mismatched_rows += 1
            
            # Check if it's a repeated header
            if row == header:
                print(f"Row {i} is a repeated header.")
                repeated_headers += 1

        print(f"\nTotal data rows: {row_count}")
        print(f"Rows with mismatched columns: {mismatched_rows}")
        print(f"Repeated headers: {repeated_headers}")

except Exception as e:
    print(f"Error reading file: {e}")
