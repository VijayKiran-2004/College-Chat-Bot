import csv
import re

input_file = r"c:\college-buddy - Copy\logs\response_log.csv"
output_file = r"c:\college-buddy - Copy\logs\response_log_cleaned.csv"

def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

try:
    with open(input_file, 'r', encoding='utf-8', newline='') as infile, \
         open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile, quoting=csv.QUOTE_MINIMAL)
        
        # Write standard header
        header = ["Timestamp", "User Query", "Bot Response", "Time Taken (s)"]
        writer.writerow(header)
        
        processed_count = 0
        skipped_header_count = 0
        
        # Skip original header if it exists and looks like a header
        first_row = True
        
        for row in reader:
            if not row:
                continue
                
            # Naive check for header row to skip repeated headers
            if "User Query" in row and "Bot Response" in row:
                skipped_header_count += 1
                continue

            # Basic structure expectation:
            # 0: Timestamp (or empty)
            # 1: Query
            # ... Middle: Response parts ...
            # -1: Time Taken (should be float-like)
            
            if len(row) < 3:
                # Too short to contain meaningful data?
                # Maybe query + response + time?
                # If only query and response, we might miss time or timestamp.
                print(f"Skipping short row: {row}")
                continue

            timestamp = row[0].strip()
            query = row[1].strip()
            
            # Find the time_taken. usually the last column.
            # We iterate from the end to find the first float-like value.
            time_taken = "0.0"
            response_end_index = len(row) - 1
            
            if is_float(row[-1]):
                time_taken = row[-1]
                response_end_index = len(row) - 1
            elif is_float(row[-2]): # Maybe there's a trailing empty column?
                 time_taken = row[-2]
                 response_end_index = len(row) - 2
            else:
                # Fallback: assume last column is Part of response, time is 0?
                # Or maybe the row is just broken.
                # Let's assume 0 if we can't find it
                 pass

            # Combine response parts
            # Response starts at index 2
            # Ends before response_end_index
            response_parts = row[2:response_end_index]
            
            # If the row was just [Timestamp, Query, Response, Time], index 2 to 3 (exclusive) -> row[2]
            # If split: [Timestamp, Query, RespPart1, RespPart2, Time] -> 2 to 4 -> row[2], row[3]
            
            full_response = ",".join(response_parts) # Join with comma if it was split by comma?
            # Actually, if csv.reader split it, it was because of a delimiter.
            # If the original text had comma, it should have been quoted.
            # If it wasn't quoted, it got split. Re-joining with comma restores it loosely.
            
            writer.writerow([timestamp, query, full_response, time_taken])
            processed_count += 1
            
        print(f"Processed {processed_count} rows.")
        print(f"Skipped {skipped_header_count} header rows.")
        print(f"Cleaned file written to: {output_file}")

except Exception as e:
    print(f"An error occurred: {e}")
