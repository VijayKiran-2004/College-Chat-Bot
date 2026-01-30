
import sys
import os
import importlib
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(os.getcwd())))

# Fix for WinError 1114
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    pass

modules_to_check = [
    # Core Services
    "app.services.intent_detector",
    "app.services.prompt_construction",
    "app.services.chain",
    "app.services.sql_system",
    "app.services.ultra_rag",
    "app.services.query_router",
    # Main Application
    "backend",
    "terminal_chat"
]

unsafe_scripts = [
    # Scripts (often have side effects like stdout rewrapping)
    "scripts/cleanup_database.py",
    "scripts/corpus_converter.py",
    "scripts/ingest.py",
    "scripts/scrape.py",
    "scripts/setup_student_database.py",
    "scripts/sql.py",
    # Tests
    "test_gender.py",
    "test_llm_sql.py",
    "test_router_upgrade.py",
    "test_typo.py"
]

print("="*60)
print("COMPREHENSIVE CODEBASE HEALTH CHECK")
print("="*60 + "\n")

failed = []
passed = []

# 1. Import Check (Safe Core Modules)
print("--- Import Checks (Core Systems) ---")
for module_name in modules_to_check:
    print(f"Checking {module_name}...", end=" ", flush=True)
    try:
        importlib.import_module(module_name)
        print("[OK]")
        passed.append(module_name)
    except Exception as e:
        print(f"[FAILED]")
        print(f"  Error: {e}")
        failed.append((module_name, str(e)))


print("\n--- Syntax Checks (No Execution) ---")
for script_path in unsafe_scripts:
    print(f"Checking {script_path}...", end=" ", flush=True)
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        compile(content, script_path, 'exec')
        print("[OK]")
        passed.append(script_path)
    except Exception as e:
        print(f"[FAILED]")
        print(f"  Error: {e}")
        failed.append((script_path, str(e)))

print("\n" + "="*60)
print(f"SUMMARY: {len(passed)} Passed, {len(failed)} Failed")
print("="*60)

if failed:
    print("\nFailures:")
    for name, err in failed:
        print(f"- {name}: {err}")
    sys.exit(1)
else:
    print("\nAll 18 files verified successfully! 🚀")
    sys.exit(0)
