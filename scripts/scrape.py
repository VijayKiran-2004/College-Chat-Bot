"""
Hybrid Web Scraper (List-Based)
Scrapes specific links from ACP-links.csv using Selenium parallel processing.
- Producer: Reads URLs from data/rawdata/ACP-links.csv
- Consumer: Selenium extraction
"""

import threading
import queue
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import json
import time
import os
import csv
import sys
from urllib.parse import urlparse

# Configuration
INPUT_CSV = "data/rawdata/ACP-links.csv"
OUTPUT_FILE = "app/database/vectordb/scraped_data.jsonl"
MAX_WORKERS = 4  # Number of concurrent threads

# Shared State
url_queue = queue.Queue()
lock = threading.Lock()

# Fix Windows encoding
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def setup_driver():
    """Setup headless Chrome driver"""
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run invisible
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--log-level=3")  # Suppress logs
    
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=chrome_options)

def load_urls_from_csv():
    """Load URLs from CSV file"""
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Error: Input file not found: {INPUT_CSV}")
        return
    
    count = 0
    try:
        with open(INPUT_CSV, 'r', encoding='utf-8') as f:
            # Check if has header or just raw list
            first_line = f.readline().strip()
            f.seek(0)
            
            # Simple reader - assumes first column is URL if CSV, or just lines
            reader = csv.reader(f)
            for row in reader:
                if not row: continue
                url = row[0].strip()
                
                # Basic validation
                if url.startswith('http'):
                    url_queue.put(url)
                    count += 1
                    
        print(f"✓ Loaded {count} URLs from {INPUT_CSV}")
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")

def extract_content(url):
    """Use Selenium to render and extract text"""
    driver = None
    try:
        driver = setup_driver()
        driver.get(url)
        
        # Wait slightly for JS
        time.sleep(2)
        
        # Extract text
        body_text = driver.find_element("tag name", "body").text
        title = driver.title
        
        # Save data
        doc = {
            "id": url,
            "title": title,
            "contents": body_text,
            "source": url,
            "type": "scraped_webpage"
        }
        
        with lock:
            with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                f.write(json.dumps(doc) + "\n")
            print(f"✓ [EXTRACTED] {title[:30]}... ({len(body_text)} chars)")
            
    except Exception as e:
        print(f"[EXTRACT ERROR] {url}: {e}")
    finally:
        if driver:
            driver.quit()

def worker():
    """Worker thread that processes URLs"""
    while True:
        try:
            url = url_queue.get(timeout=3) # Wait short time since queue is pre-filled
        except queue.Empty:
            return # Exit if empty

        extract_content(url)
        url_queue.task_done()

if __name__ == "__main__":
    print("="*70)
    print("HYBRID SCRAPER (CSV LIST MODE)")
    print("="*70)
    
    # Ensure output dir
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # Clear old file
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    # Load URLs
    load_urls_from_csv()
    
    if url_queue.empty():
        print("No URLs to scrape. Exiting.")
        sys.exit(0)

    print(f"\nStarting {MAX_WORKERS} worker threads...")
    
    threads = []
    for _ in range(MAX_WORKERS):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)
    
    # Wait for threads
    for t in threads:
        t.join()

    print("\n" + "="*70)
    print("SCRAPING COMPLETE")
    print(f"Data saved to: {OUTPUT_FILE}")
    print("="*70)
