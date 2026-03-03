# TKRCET Scraper Instructions

This folder now contains an optimized web scraper `tkrcet_scraper.py` that uses Playwright for dynamic rendering and Groq for data structuring.

## Prerequisites

1. **Install Dependencies**:
   Ensure you have the latest requirements installed.
   ```powershell
   pip install -r requirements.txt
   ```

2. **Setup Playwright**:
   Download the required browser binaries.
   ```powershell
   playwright install chromium
   ```

3. **Input File**:
   Create a file named `tkrcet_links.txt` in the project root. Add the URLs you want to scrape, one per line.
   Example:
   ```text
   https://tkrcet.ac.in/about-us/
   https://tkrcet.ac.in/academics/departments/cse/
   ```

4. **API Key**:
   Set your Groq API key in your environment for security.
   ```powershell
   $env:GROQ_API_KEY="your_actual_key_here"
   ```

## Running the Scraper

Run the following command from the project root:
```powershell
python scripts/tkrcet_scraper.py
```

## Outputs

- **`outputs/`**: This folder will contain individual `.json` files for every URL scraped.
- **`all_results.json`**: A master file containing all scraped data in a single JSON array, ready for ingestion into the chatbot.

## Logic Improvements Made
- **Performance**: Used a single browser session instead of restarting for every URL.
- **Data Quality**: Enhanced JavaScript traversal prevents duplicated text blocks.
- **Reliability**: Used Groq's high-speed Llama 3.1 8B model with **JSON Mode** enabled.
- **Safety**: Added rate-limiting (1.5s) to avoid API bans.
