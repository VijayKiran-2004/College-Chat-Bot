"""
Detailed Log Analysis Script
Analyzes response logs to identify patterns and issues
"""
import pandas as pd

# Read the log file
df = pd.read_excel('logs/response_log.xlsx')

print("=" * 80)
print("DETAILED LOG ANALYSIS")
print("=" * 80)
print()

# Overall statistics
print(f"Total queries: {len(df)}")
print(f"Average response time: {df['Time Taken (s)'].mean():.2f}s")
print(f"Median response time: {df['Time Taken (s)'].median():.2f}s")
print()

# Categorize by response time
fast = df[df['Time Taken (s)'] < 1]
moderate = df[(df['Time Taken (s)'] >= 1) & (df['Time Taken (s)'] < 10)]
slow = df[(df['Time Taken (s)'] >= 10) & (df['Time Taken (s)'] < 50)]
very_slow = df[df['Time Taken (s)'] >= 50]

print("Response Time Distribution:")
print(f"  Fast (<1s):        {len(fast)} queries ({len(fast)/len(df)*100:.1f}%)")
print(f"  Moderate (1-10s):  {len(moderate)} queries ({len(moderate)/len(df)*100:.1f}%)")
print(f"  Slow (10-50s):     {len(slow)} queries ({len(slow)/len(df)*100:.1f}%)")
print(f"  Very Slow (>50s):  {len(very_slow)} queries ({len(very_slow)/len(df)*100:.1f}%)")
print()

# Analyze by source
print("Performance by Source:")
for source in df['Source'].unique():
    source_df = df[df['Source'] == source]
    print(f"  {source}:")
    print(f"    Count: {len(source_df)}")
    print(f"    Avg time: {source_df['Time Taken (s)'].mean():.2f}s")
    print(f"    Max time: {source_df['Time Taken (s)'].max():.2f}s")
print()

# Identify problematic queries
print("=" * 80)
print("PROBLEMATIC QUERIES (Should be fast but aren't)")
print("=" * 80)
print()

# KB queries that should be instant
kb_slow = df[(df['Source'] == 'RAG/Knowledge Base') & (df['Time Taken (s)'] > 10)]
if len(kb_slow) > 0:
    print("KB queries taking >10s (should be instant):")
    for idx, row in kb_slow.iterrows():
        print(f"  {row['Time Taken (s)']:.2f}s - {row['User Query']}")
    print()

# System timeouts
system_timeouts = df[df['Source'] == 'System']
if len(system_timeouts) > 0:
    print("System timeouts (queries that hit 70s limit):")
    for idx, row in system_timeouts.iterrows():
        print(f"  {row['Time Taken (s)']:.2f}s - {row['User Query']}")
    print()

# Queries that should hit KB but went to RAG
kb_keywords = ['timings', 'address', 'location', 'principal', 'hod', 'chairman', 
               'fee', 'transport', 'canteen', 'ncc', 'nss']

print("Queries with KB keywords that went to RAG/took long:")
for idx, row in df.iterrows():
    query_lower = row['User Query'].lower()
    if any(keyword in query_lower for keyword in kb_keywords):
        if row['Time Taken (s)'] > 5:
            print(f"  {row['Time Taken (s)']:.2f}s - {row['User Query']} [{row['Source']}]")

print()
print("=" * 80)
