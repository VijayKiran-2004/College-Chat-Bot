import pandas as pd

# Read the log file
df = pd.read_excel('logs/response_log.xlsx')

# Write to output file
with open('log_analysis.txt', 'w', encoding='utf-8') as f:
    f.write(f"Total queries logged: {len(df)}\n\n")
    f.write("=" * 80 + "\n")
    
    # Find slow queries (> 50 seconds)
    slow = df[df['Time Taken (s)'] > 50].sort_values('Time Taken (s)', ascending=False)
    
    f.write(f"\nQueries taking > 50 seconds: {len(slow)} queries\n\n")
    f.write("=" * 80 + "\n")
    
    for idx, row in slow.head(15).iterrows():
        f.write(f"\nTime: {row['Time Taken (s)']:.2f}s\n")
        f.write(f"Query: {row['User Query']}\n")
        f.write(f"Source: {row['Source']}\n")
        f.write(f"Timestamp: {row['Timestamp']}\n")
        f.write("-" * 80 + "\n")
    
    # Statistics
    if len(slow) > 0:
        f.write(f"\n\nStatistics for slow queries:\n")
        f.write(f"Average time: {slow['Time Taken (s)'].mean():.2f}s\n")
        f.write(f"Max time: {slow['Time Taken (s)'].max():.2f}s\n")
        f.write(f"Min time: {slow['Time Taken (s)'].min():.2f}s\n")
    
    # Show all query times
    f.write("\n\nAll query times:\n")
    f.write(df[['Timestamp', 'User Query', 'Time Taken (s)', 'Source']].to_string())

print("Analysis saved to log_analysis.txt")
