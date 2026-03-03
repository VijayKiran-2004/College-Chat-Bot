import asyncio
from playwright.async_api import async_playwright
from groq import AsyncGroq
import json
import re
import os
from pathlib import Path

# =============================================================
# CONFIG
# =============================================================

# Always keep your keys secret! Set this in your environment:
# PowerShell: $env:GROQ_API_KEY="your_key"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set. Please set it to proceed.")

client = AsyncGroq(api_key=GROQ_API_KEY)

INPUT_FILE = "tkrcet_links.txt"   # List of TKRCET URLs
OUTPUT_DIR = "outputs"            # Folder for per-page JSON
COMBINED_OUTPUT = "all_results.json"

# =============================================================
# PLAYWRIGHT: Extract clean visible text
# =============================================================

async def extract_text(browser, url):
    """Navigates to URL and extracts clean, structured text using a shared browser instance."""
    page = await browser.new_page()
    try:
        print(f"\n🌐 Loading: {url}")
        # Using domcontentloaded for better reliability on dynamic sites
        await page.goto(url, wait_until="domcontentloaded", timeout=45000)

        # Scroll to load lazy content (images, dynamic tables, etc.)
        await page.evaluate("""
            async () => {
                let total = 0;
                while (total < document.body.scrollHeight) {
                    window.scrollBy(0, 800);
                    total += 800;
                    await new Promise(r => setTimeout(r, 120));
                }
            }
        """)

        print("📄 Extracting visible text...")

        # Optimized Extraction Logic: Prevents text duplication by checking tags
        text = await page.evaluate("""
            () => {
                function extract(el) {
                    let tag = el.tagName?.toLowerCase() || "";
                    // Skip noise elements
                    if (["script","style","nav","footer","header","svg","img", "noscript", "aside", "embed", "object"].includes(tag))
                        return "";

                    // If we grab innerText of these blocks, DO NOT process children separately
                    if (["h1","h2","h3","h4"].includes(tag)) return "\\n# " + el.innerText.trim() + "\\n";
                    if (["p", "article"].includes(tag)) return "\\n" + el.innerText.trim() + "\\n";
                    if (tag === "li") return "- " + el.innerText.trim() + "\\n";
                    if (tag === "table") return "\\n[TABLE]\\n" + el.innerText.trim() + "\\n[/TABLE]\\n";

                    let out = "";
                    for (let c of el.children) {
                        out += extract(c);
                    }
                    return out;
                }
                return extract(document.body);
            }
        """)
        return text.strip()
    except Exception as e:
        print(f"❌ Playwright error on {url}: {e}")
        return ""
    finally:
        await page.close()

# =============================================================
# GROQ: Convert clean text → structured JSON
# =============================================================

async def extract_json(clean_text):
    """Sends cleaned text to Groq LLM to convert into structured JSON format."""
    if not clean_text or len(clean_text) < 50:
        return {"sections": []}

    # Model truncated context window strategy
    prompt = f"""
Convert the following CLEAN TEXT from the TKRCET college website into structured JSON.

Rules:
- Headings start with "#"
- Lists start with "-"
- Tables appear as [TABLE] ... [/TABLE]
- Preserve original order and factual accuracy
- Output a JSON object with a "sections" array containing "title" and "content" fields.

TEXT:
{clean_text[:15000]}
"""

    print("⚡ Sending to Groq (JSON Mode)...")
    try:
        # Native JSON mode for reliability
        response = await client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )
        print("✅ Groq responded successfully.")
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"⚠️ Groq Processing Error: {e}")
        return {"sections": [], "error": str(e)}

# =============================================================
# HELPER: Convert URL → safe file name
# =============================================================

def sanitize_filename(url):
    """Creates a Windows-safe filename from a URL."""
    clean = re.sub(r"https?://", "", url)
    # Replaces all illegal Windows characters: < > : " / \ | ? *
    clean = re.sub(r'[<>:"/\\|?*]', '_', clean)
    return clean[:100] + ".json"

# =============================================================
# MAIN CRAWLER
# =============================================================

async def main():
    # Setup directories
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found in current directory.")
        print(f"Please create {INPUT_FILE} and add URLs (one per line).")
        return

    # Load and filter URLs
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw_lines = [line.strip() for line in f if line.strip()]

    # Extract first URL found in each line to handle "URL PageName" format
    urls = []
    url_pattern = re.compile(r'https?://[^\s,]+')
    for line in raw_lines:
        match = url_pattern.search(line)
        if match:
            urls.append(match.group())

    final_urls = [
        u for u in urls if not any(ext in u.lower() for ext in [".pdf", ".jpg", ".png", ".zip", ".docx"])
        and "#pdf" not in u.lower()
    ]

    print(f"\n🔗 URLs to process: {len(final_urls)}")
    combined_results = []

    # Initialize Browser ONCE for the whole session
    async with async_playwright() as p:
        print("🚀 Starting Chromium...")
        browser = await p.chromium.launch(headless=True)
        
        for url in final_urls:
            try:
                # 1. Extraction
                text = await extract_text(browser, url)
                
                # 2. Structuring
                json_data = await extract_json(text)

                # 3. Save Individual File
                filename = sanitize_filename(url)
                filepath = os.path.join(OUTPUT_DIR, filename)

                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)

                print(f"💾 Saved JSON: {filepath}")

                combined_results.append({
                    "url": url,
                    "data": json_data
                })

                # 4. Rate Limiting (1.5s delay to protect API limits)
                await asyncio.sleep(1.5)

            except Exception as e:
                print(f"❌ Global Error for {url}: {e}")
                combined_results.append({"url": url, "error": str(e)})

        await browser.close()

    # Save final combined results
    with open(COMBINED_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=2, ensure_ascii=False)

    print("\n" + "="*50)
    print("🎉 SCRAPING COMPLETE")
    print(f"Total processed: {len(combined_results)}")
    print(f"Individual files: ./{OUTPUT_DIR}/")
    print(f"Master file: {COMBINED_OUTPUT}")
    print("="*50 + "\n")

# =============================================================

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
