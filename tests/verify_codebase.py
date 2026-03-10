import requests
import sqlite3
import subprocess
from pathlib import Path


def check_memory():
    print("--- Checking System Memory ---")
    try:
        # PowerShell command to get memory in MB
        cmd = (
            "Get-CimInstance Win32_OperatingSystem |"
            "Select-Object TotalVisibleMemorySize, FreePhysicalMemory"
        )
        output = subprocess.check_output(["powershell", "-Command", cmd], text=True)
        lines = output.strip().split("\n")
        if len(lines) >= 3:
            memory_data = lines[2].split()
            total_kb = int(memory_data[0])
            free_kb = int(memory_data[1])

            total_gb = round(total_kb / (1024 * 1024), 2)
            free_gb = round(free_kb / (1024 * 1024), 2)

            print(f"Total Memory: {total_gb} GB")
            print(f"Free Memory:  {free_gb} GB")

            if free_gb < 1.0:
                print("[WARNING] Very low free memory! Ollama might fail.")
            elif free_gb < 3.0:
                print(
                    "[NOTE] Free memory is tight. "
                    "Close heavy apps like Chrome for best performance."
                )
            else:
                print("[OK] Sufficient memory available.")
    except Exception as e:
        print(f"[FAILED] Could not check memory: {e}")
    print()


def check_ollama():
    print("--- Checking Ollama ---")
    url = "http://127.0.0.1:11434/api/tags"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print("[OK] Ollama is running.")
            models = [m["name"] for m in response.json().get("models", [])]
            required_model = "llama3.2:3b"
            if any(required_model in m for m in models):
                print(f"[OK] Model '{required_model}' found.")
            else:
                print(f"[FAILED] Model '{required_model}' NOT found.")
                print(f"       Please run: ollama pull {required_model}")
        else:
            print(f"[FAILED] Ollama returned status {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("[FAILED] Ollama is NOT running. Please start it with 'ollama serve'.")
    except Exception as e:
        print(f"[FAILED] Error checking Ollama: {e}")
    print()


def check_files():
    print("--- Checking Data Files ---")
    project_root = Path(__file__).parent.parent  # Go up from tests/ to project root
    files_to_check = [
        ("Knowledge Base", "data/knowledge_base.json"),
        ("RAG Corpus", "data/chunks/corpus_ultrarag.jsonl"),
        ("Student DB", "app/database/students.db"),
    ]

    for name, rel_path in files_to_check:
        full_path = project_root / rel_path
        if full_path.exists():
            print(f"[OK] {name} found at {rel_path}")
            if rel_path.endswith(".db"):
                try:
                    conn = sqlite3.connect(full_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT count(*) FROM students")
                    count = cursor.fetchone()[0]
                    print(f"     (Contains {count} students)")
                    conn.close()
                except Exception as e:
                    print(f"     [FAILED] Could not read database: {e}")
        else:
            print(f"[FAILED] {name} NOT found at {rel_path}")
            if "corpus" in rel_path:
                print(
                    "         Run 'python scripts/tkrcet_scraper.py',"
                    "then 'generate_vectors.py', 'corpus_converter.py', and 'ingest.py'"
                )
            elif "students.db" in rel_path:
                print(
                    "         The students.db file is missing."
                    " Please check if you have a backup or need to regenerate it."
                )
    print()


def check_env():
    print("--- Checking Environment ---")
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        print("[OK] .env file found.")
        with open(env_path, "r") as f:
            content = f.read()
            if "GROQ_API_KEY" in content and "=" in content:
                key = content.split("GROQ_API_KEY=")[1].split("\n")[0].strip()
                if key and not key.startswith("your_"):
                    print("[OK] GROQ_API_KEY is set.")
                else:
                    print("[FAILED] GROQ_API_KEY is empty or default.")
            else:
                print("[FAILED] GROQ_API_KEY not found in .env")
    else:
        print(
            "[FAILED] .env file NOT found."
            "Run 'copy .env.example .env' and add your key."
        )
    print()


if __name__ == "__main__":
    print("========================================")
    print(" COLLEGE BUDDY - ENVIRONMENT DIAGNOSTIC ")
    print("========================================\n")
    check_ollama()
    check_memory()
    check_files()
    check_env()
    print("========================================\n")
