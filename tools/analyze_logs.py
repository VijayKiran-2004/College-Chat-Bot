"""
tools/analyze_logs.py — Production Log Analyser
Reads logs/response_log.xlsx (Production sheet) and prints performance stats.
Run: python tools/analyze_logs.py
"""
import pandas as pd
from pathlib import Path

LOG = Path("logs/response_log.xlsx")

# Production sheet columns: Timestamp | User Query | Bot Response |
#                            Time Taken (s) | Session ID | Source
TIME_COL   = "Time Taken (s)"
QUERY_COL  = "User Query"
SOURCE_COL = "Source"


def main():
    if not LOG.exists():
        print("Log file not found. Run refresh_logs.py first.")
        return

    df = pd.read_excel(LOG, sheet_name="Production")

    if df.empty:
        print("Production sheet is empty — no queries logged yet.")
        return

    print("=" * 70)
    print("PRODUCTION LOG ANALYSIS")
    print("=" * 70)
    print(f"Total queries logged : {len(df)}")

    if TIME_COL in df.columns:
        df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")
        print(f"Average response time: {df[TIME_COL].mean():.2f}s")
        print(f"Median response time : {df[TIME_COL].median():.2f}s")
        print(f"Max response time    : {df[TIME_COL].max():.2f}s")
        print()

        # Response-time distribution
        fast      = df[df[TIME_COL] < 1]
        moderate  = df[(df[TIME_COL] >= 1)  & (df[TIME_COL] < 10)]
        slow      = df[(df[TIME_COL] >= 10) & (df[TIME_COL] < 50)]
        very_slow = df[df[TIME_COL] >= 50]
        n = len(df)
        print("Response Time Distribution:")
        print(f"  Fast     (< 1 s)  : {len(fast):4d}  ({len(fast)/n*100:.1f}%)")
        print(f"  Moderate (1–10 s) : {len(moderate):4d}  ({len(moderate)/n*100:.1f}%)")
        print(f"  Slow     (10–50s) : {len(slow):4d}  ({len(slow)/n*100:.1f}%)")
        print(f"  Very Slow (>50s)  : {len(very_slow):4d}  ({len(very_slow)/n*100:.1f}%)")
        print()

    # Per-source breakdown
    if SOURCE_COL in df.columns:
        print("Performance by Source:")
        for source in df[SOURCE_COL].dropna().unique():
            sdf = df[df[SOURCE_COL] == source]
            avg = sdf[TIME_COL].mean() if TIME_COL in df.columns else float("nan")
            print(f"  {source:<20} count={len(sdf):4d}  avg={avg:.2f}s")
        print()

    # Recent 5 queries
    print("Last 5 queries:")
    cols = [c for c in [QUERY_COL, SOURCE_COL, TIME_COL] if c in df.columns]
    print(df[cols].tail(5).to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    main()
