import re
import os
import time
import random
import json
import warnings
import fitz
import requests
from pathlib import Path
from dotenv import load_dotenv
from scrapling import StealthyFetcher
from groq import Groq
import undetected_chromedriver as uc
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

from playwright.sync_api import sync_playwright

# Suppress scrapling's internal deprecation warning (harmless — "has no effect")
warnings.filterwarnings("ignore", message=".*deprecated.*", module="scrapling.*")
# Suppress Playwright's noisy cleanup warning when browser is force-closed
warnings.filterwarnings("ignore", message=".*TargetClosedError.*")


# =============================================================
# CONFIG
# =============================================================

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
load_dotenv(_PROJECT_ROOT / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not set in .env")

client = Groq(api_key=GROQ_API_KEY)

# Absolute paths
INPUT_FILE = str(_PROJECT_ROOT / "scripts" / "tkrcet_links.txt")
OUTPUT_DIR = str(_PROJECT_ROOT / "data" / "scraped_data" / "outputs")
COMBINED_OUTPUT = str(_PROJECT_ROOT / "data" / "scraped_data" / "all_results.json")

# Initialize Scrapling StealthyFetcher with Playwright for maximum stealth
fetcher = StealthyFetcher()

# =============================================================
# EXTRACTION: Scrapling + Groq (Sync)
# =============================================================


def _fetch_pdf_bytes_direct(url, referer):
    """Tries to fetch PDF bytes via a Playwright browser session.
    Returns bytes or None."""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36",
                viewport={"width": 1280, "height": 800},
                locale="en-US",
            )
            page = context.new_page()
            try:
                page.goto(referer, wait_until="domcontentloaded", timeout=20000)
                time.sleep(random.uniform(1.5, 3.0))
            except Exception:
                pass  # best-effort cookie pickup
            response = page.request.get(
                url,
                headers={
                    "Referer": referer,
                    "Accept": "application/pdf,application/octet-stream,*/*;q=0.8",
                },
                timeout=60000,
            )
            pdf_body = response.body() if response.ok else None
            status = response.status
            browser.close()
        if pdf_body:
            return pdf_body
        print(f"   ↻ Direct fetch got {status} — trying Wayback Machine...")
        return None
    except Exception as e:
        # Playwright may throw 'Event loop is closed' after a UC fallback ran;
        # treat as a soft failure and let Wayback Machine take over.
        print(f"   ↻ Direct fetch error: {e}")
        return None


def _fetch_pdf_bytes_wayback(url):
    """Fetches a PDF from the Wayback Machine archive.
    Public college documents are almost always archived and served without
    bot protection.
    Returns bytes on success, None if not archived or on error.
    """
    print(f"   🗄️  Checking Wayback Machine for: {url}")
    try:
        # Ask the Wayback availability API for the closest snapshot
        meta = requests.get(
            f"https://archive.org/wayback/available?url={url}", timeout=10
        ).json()
        snapshot = meta.get("archived_snapshots", {}).get("closest", {})
        if not snapshot.get("available"):
            print("   ❌ No Wayback Machine snapshot found.")
            return None

        # Build the raw-content URL: insert 'if_' after the timestamp to skip the
        # toolbar
        wb_url = snapshot[
            "url"
        ]  # e.g. https://web.archive.org/web/20240901120000/https://...
        parts = wb_url.split("/web/")
        if len(parts) == 2:
            timestamp = parts[1].split("/")[0]
            raw_url = f"https://web.archive.org/web/{timestamp}if_/{url}"
        else:
            raw_url = wb_url

        resp = requests.get(raw_url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 200 and resp.content:
            print(f"   ✅ Got {len(resp.content):,} bytes from Wayback Machine.")
            return resp.content
        print(f"   ❌ Wayback raw fetch failed (Status {resp.status_code}).")
        return None
    except Exception as e:
        print(f"   ❌ Wayback Machine error: {e}")
        return None


def _extract_text_from_pdf_bytes(pdf_bytes):
    """Parses PDF bytes with PyMuPDF and returns extracted text (first 15 pages)."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = "".join(pg.get_text() for pg in doc[:15])
    doc.close()
    return text.strip()


def process_pdf_sync(url, referer=None):
    """Extracts text from a PDF URL using a two-stage fallback strategy.

    Stage 1: Playwright browser session (real TLS fingerprint + site cookies)
    Stage 2: Wayback Machine archive (bypasses server bot-blocking entirely)
    """
    from urllib.parse import urlparse

    effective_referer = referer or f"{urlparse(url).scheme}://{urlparse(url).netloc}/"
    print(f"\n📄 Processing PDF: {url}")

    # Stage 1 — Playwright direct
    pdf_bytes = _fetch_pdf_bytes_direct(url, effective_referer)

    # Stage 2 — Wayback Machine
    if not pdf_bytes:
        pdf_bytes = _fetch_pdf_bytes_wayback(url)

    if not pdf_bytes:
        print("❌ Could not retrieve PDF from any source.")
        return None

    try:
        text = _extract_text_from_pdf_bytes(pdf_bytes)
        if not text:
            print("⚠️ No extractable text in PDF (may be scanned/image-only).")
            return None
        print(f"✅ Extracted {len(text):,} characters from PDF.")
        return extract_json_sync(text)
    except Exception as e:
        print(f"❌ PDF parse error: {e}")
        return None


def _detect_chrome_version():
    """Reads the installed Chrome major version, trying multiple registry paths.
    Returns the int major version (e.g. 145), or None to let UC auto-detect.
    Registry paths checked (covers system-wide and user installs on 32/64-bit):
    """
    import subprocess as _sp

    _reg_paths = [
        r"HKLM\SOFTWARE\Google\Chrome\BLBeacon",
        r"HKLM\SOFTWARE\WOW6432Node\Google\Chrome\BLBeacon",
        r"HKCU\SOFTWARE\Google\Chrome\BLBeacon",
    ]
    for path in _reg_paths:
        try:
            result = _sp.run(
                ["reg", "query", path, "/v", "version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            match = re.search(r"version\s+REG_SZ\s+(\d+)", result.stdout)
            if match:
                ver = int(match.group(1))
                print(f"   🥙 Detected Chrome v{ver} from registry.")
                return ver
        except Exception:
            continue
    print(
        "   ⚠️ Could not detect Chrome version from registry — letting UC auto-detect."
    )
    return None  # UC will try to auto-detect


def fetch_with_uc_fallback(url):
    """Fetches a page using undetected-chromedriver to bypass Cloudflare on 403 pages.

    Undetected ChromeDriver (UC) patches the binary-level webdriver flag that
    Cloudflare checks — it appears as a completely normal browser session.
    Only used as a fallback when StealthyFetcher returns 403.
    """
    print(f"🔓 UC Fallback: trying undetected-chromedriver for {url}")
    driver = None
    try:
        options = uc.ChromeOptions()
        options.add_argument("--headless=new")  # new headless keeps UC patches
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1280,800")
        options.add_argument(
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )

        driver = uc.Chrome(
            options=options, use_subprocess=True, version_main=_detect_chrome_version()
        )
        driver.get(url)

        # Wait up to 15s for the body to be present (handles JS challenges)
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        time.sleep(random.uniform(2.0, 4.0))  # let any JS finish rendering

        page_text = driver.find_element(By.TAG_NAME, "body").text.strip()
        driver.quit()

        # Reject if the page is a security/firewall block page, not real content
        _block_signals = [
            "malcare firewall",
            "blocked because",
            "just a moment",
            "access denied",
            "checking your browser",
            "enable javascript",
        ]
        if (
            not page_text
            or len(page_text) < 50
            or any(sig in page_text.lower() for sig in _block_signals)
        ):
            print(
                "⚠️UC got a firewall/bot-challenge page," "not real content — skipping."
            )
            return None

        print(f"✅ UC Fallback succeeded — extracted {len(page_text)} characters.")
        return extract_json_sync(page_text)

    except Exception as e:
        print(f"❌ UC Fallback error for {url}: {e}")
        if driver:
            try:
                driver.quit()
            except Exception:
                pass
        return None


def process_url(url):
    """Fetches content using Scrapling (web) or requests (pdf)
    and structures it using Groq."""
    try:
        # Check if URL is a PDF
        if url.lower().endswith(".pdf") or "pdf" in url.lower():
            # Pass the URL itself as referer — PDFs are direct links so we use the
            # parent domain as a proxy for the page that would have linked to them.
            from urllib.parse import urlparse

            parsed = urlparse(url)
            parent_url = f"{parsed.scheme}://{parsed.netloc}/"
            return process_pdf_sync(url, referer=parent_url)

        print(f"\n🌐 Loading: {url}")
        # Add human-like randomized sleep
        time.sleep(random.uniform(3.0, 7.0))

        # Headers to look like a real user
        fetch_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Referer": "https://www.google.com/search?q=tkr+college+engineering+and+technology",
            "Accept-Language": "en-US,en;q=0.9",
        }

        # Primary: Playwright-based StealthyFetcher
        page_data = fetcher.fetch(url, headers=fetch_headers)

        if page_data.status == 403:
            print(
                "⚠️ StealthyFetcher got 403 — trying undetected-chromedriver fallback..."
            )
            return fetch_with_uc_fallback(url)

        print("📄 Extracting visible text...")
        # SCRAPING FIX: Use CSS selectors to get text if page_data.text is empty
        clean_text = page_data.text.strip()

        if not clean_text or len(clean_text) < 100:
            # Fallback: Get all text from body while ignoring scripts and styles
            try:
                all_text_nodes = page_data.css(
                    "body *:not(script):not(style)::text"
                ).getall()
                clean_text = " ".join([t.strip() for t in all_text_nodes if t.strip()])
            except Exception as e:
                print(f"⚠️ Manual extraction failed: {e}")

        if len(clean_text) < 50:
            print("⚠️ Insufficient content found.")
            return {"sections": []}

        # FIREWALL DETECTION: MalCare & similar return HTTP 200 with a fake block page.
        # Detect by known phrases and skip — don't waste a Groq call on garbage.
        _firewall_signals = [
            "malcare firewall",
            "blocked because of malicious",
            "blocked because of suspicious",
            "reference id:",  # MalCare block always has a Reference ID
            "just a moment",  # Cloudflare JS challenge
            "checking your browser",
            "enable javascript and cookies",
            "access denied",
        ]
        lower_text = clean_text.lower()
        if (
            any(sig in lower_text for sig in _firewall_signals)
            and len(clean_text) < 2000
        ):
            print(
                f"🚫 Firewall block page detected (HTTP 200 disguised) — skipping {url}"
            )
            return None

        # Structure via Groq (Sync)
        structured_json = extract_json_sync(clean_text)
        return structured_json

    except Exception as e:
        print(f"❌ Error on {url}: {e}")
        return None


def extract_json_sync(clean_text):
    """Sends cleaned text to Groq LLM (Sync) to convert into structured JSON format."""
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
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        print("✅ Groq responded successfully.")
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"⚠️ Groq Processing Error: {e}")
        return {"sections": [], "error": str(e)}


def sanitize_filename(url):
    """Creates a Windows-safe filename from a URL."""
    clean = re.sub(r"https?://", "", url)
    clean = re.sub(r'[<>:"/\\|?*]', "_", clean).strip("_")
    return clean[:100] + ".json"


# =============================================================
# MAIN CRAWLER (Sync)
# =============================================================


def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found.")
        return

    # Load and filter URLs — skip blank lines and comment lines (starting with #)
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw_lines = [line.strip() for line in f if line.strip()]

    urls = []
    url_pattern = re.compile(r"https?://[^\s,]+")
    for line in raw_lines:
        if line.startswith("#"):  # skip commented-out URLs
            continue
        match = url_pattern.search(line)
        if match:
            urls.append(match.group())

    # PDF INCLUSION: We now allow PDFs
    final_urls = [
        u
        for u in urls
        if not any(
            ext in u.lower() for ext in [".jpg", ".png", ".zip", ".docx", ".exe"]
        )
    ]

    print(f"\n🔗 URLs to process: {len(final_urls)}")
    combined_results = []

    # BATCH SETTINGS
    BATCH_SIZE = 10
    COOLDOWN_TIME = (120, 180)  # 2-3 minutes pause between batches

    for i, url in enumerate(final_urls):
        try:
            # Batching logic: After every BATCH_SIZE items, take a long break
            if i > 0 and i % BATCH_SIZE == 0:
                wait_time = random.uniform(*COOLDOWN_TIME)
                print(
                    f"\n☕ Cooling off to evade firewall... Sleeping for {int(wait_time)}s"
                )
                time.sleep(wait_time)

            json_data = process_url(url)

            # 10-second polite pause after every URL (±2s jitter to avoid pattern detection)
            pause = random.uniform(8.0, 12.0)
            print(f"⏳ Waiting {pause:.1f}s before next URL...")
            time.sleep(pause)

            if json_data:
                filename = sanitize_filename(url)
                filepath = os.path.join(OUTPUT_DIR, filename)

                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)

                print(f"💾 Saved JSON: {filepath}")

                combined_results.append({"url": url, "data": json_data})

        except Exception as e:
            print(f"❌ Global Error for {url}: {e}")
            combined_results.append({"url": url, "error": str(e)})

    # Save final combined results
    with open(COMBINED_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 50)
    print("🎉 SCRAPING COMPLETE")
    print(f"Total processed: {len(combined_results)}")
    print(f"Master file: {COMBINED_OUTPUT}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
