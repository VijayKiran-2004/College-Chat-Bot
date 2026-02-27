import json
import os
import re
from pathlib import Path

# Configuration
INPUT_FILE = "all_results.json"
OUTPUT_FILE = "app/database/vectordb/scraped_data.jsonl"

def flatten_content(content):
    """Recursively flattens various content types (str, list, dict) into a single string."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            parts.append(flatten_content(item))
        return "\n".join(filter(None, parts))
    if isinstance(content, dict):
        # Handle nested sections or key-value pairs
        title = content.get("title", "").strip()
        body = content.get("content", "")
        text = ""
        if title:
            text += f"## {title}\n"
        text += flatten_content(body)
        return text
    return str(content)

def main():
    print("="*70)
    print("PREPARING DATA FOR INGESTION")
    print("="*70)

    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found. Did the scraper finish?")
        return

    print(f"Reading {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    print(f"Found {len(all_data)} pages. Processing...")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    count = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in all_data:
            url = entry.get("url")
            data = entry.get("data", {})
            
            if "error" in entry:
                continue

            sections = data.get("sections", [])
            if not sections:
                continue

            full_content = ""
            main_title = "Unknown"
            
            for i, section in enumerate(sections):
                title = section.get("title", "").strip()
                content = section.get("content", "")
                
                if i == 0 and title:
                    main_title = title.replace("#", "").strip()
                    
                if title:
                    full_content += f"{title}\n"
                
                full_content += flatten_content(content) + "\n\n"

            if len(full_content.strip()) < 50:
                continue

            # Create the JSONL document structure expected by the system
            doc = {
                "id": url,
                "title": main_title,
                "contents": full_content.strip(),
                "source": url,
                "type": "scraped_webpage",
                "structured": True
            }

            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            count += 1

    print(f"✓ Successfully processed {count} documents.")
    print(f"✓ Saved to: {OUTPUT_FILE}")
    print("\nNext: Run the vector generation script.")
    print("="*70)

if __name__ == "__main__":
    main()
