"""
rescrape_single.py  —  Re-scrape a single URL and overwrite its JSON output.

Usage:
    python scripts/rescrape_single.py "https://tkrcet.ac.in/computer-science-and
    -engineering/"
    python scripts/rescrape_single.py "https://tkrcet.ac.in/some-other-page/"
"""

import sys
import os
import json
import warnings
from pathlib import Path
from dotenv import load_dotenv
from tkrcet_scraper import process_url, sanitize_filename

warnings.filterwarnings("ignore", message=".*deprecated.*", module="scrapling.*")
warnings.filterwarnings("ignore", message=".*TargetClosedError.*")

# ── Project root & .env ────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

load_dotenv(_PROJECT_ROOT / ".env")

# ── Import the helpers from the main scraper ──────────────────────────────────
sys.path.insert(0, str(_SCRIPT_DIR))  # make sure scripts/ is on the path


OUTPUT_DIR = str(_PROJECT_ROOT / "data" / "scraped_data" / "outputs")


def rescrape(url: str):
    print(f"\n🎯 Re-scraping: {url}\n")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    result = process_url(url)

    if result is None:
        print(
            "\n❌ process_url returned None — page could not be fetched or was blocked."
        )
        return

    filename = sanitize_filename(url)
    filepath = os.path.join(OUTPUT_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    sections = result.get("sections", [])
    print(f"\n✅ Done!  {len(sections)} section(s) extracted.")
    print(f"💾 Saved → {filepath}")

    # Pretty-print first section as a quick sanity-check
    if sections:
        first = sections[0]
        print("\n📌 First section preview:")
        print(f"Title  : {first.get('title','(no title)')}")
        content = str(first.get("content", ""))
        print(f"   Content: {content[:200]}{'...' if len(content) > 200 else ''}")
    else:
        print("\n⚠️  No sections in the result — check the JSON for details.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default to CSE page if no argument given
        target = "https://tkrcet.ac.in/computer-science-and-engineering/"
        print(f"ℹ️  No URL argument given. Defaulting to: {target}")
    else:
        target = sys.argv[1].strip()

    rescrape(target)
