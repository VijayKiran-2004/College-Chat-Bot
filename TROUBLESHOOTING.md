# College Buddy - Troubleshooting Guide

This guide helps resolve common issues when running the College Buddy backend.

## 🛠️ Quick System Check

We added a diagnostic tool to check your environment automatically.

1. Open a terminal in the project folder.
2. Run:
   ```powershell
   python verify_codebase.py
   ```
3. Look for `[FAILED]` marks in the output.

---

## 🛑 Common Errors & Fixes

### 1. "Ollama is running in the background" / "Address already in use"
**Error:** `[WinError 10048] Only one usage of each socket address...`

**Cause:** You are trying to run `ollama serve` but Ollama is **already running** (likely in the background or system tray).

**Fix:**
*   **Don't run `ollama serve` manually** if the Ollama app is already open in your taskbar.
*   Check the System Tray (near the clock) for the Ollama icon.
*   If you need to restart it:
    1. Right-click Ollama icon > Quit.
    2. Open Task Manager > End any `ollama.exe` tasks.
    3. Run `ollama serve` again.

---

### 2. "Requirement ... conflict" / Version Errors
**Error:** `pip` gives errors about conflicting versions (e.g., numpy, torch).

**Fix:**
The unpinned dependencies (`langchain-experimental`, `langchain-huggingface`, `scikit-learn`, `transformers`) must have version pins compatible with `langchain==0.2.16`.
1. Make sure your `requirements.txt` has pinned versions (not bare package names) on lines 41-44
2. Delete your virtual environment and recreate:
   ```powershell
   Remove-Item -Recurse -Force .venv
   python -m venv .venv
   .\.venv\Scripts\Activate
   pip install -r requirements.txt
   ```

---

### 3. "Read timed out" / Ollama not responding
**Error:** `HTTPConnectionPool... Read timed out`

**Cause:** Your PC is taking too long to load the AI model into memory.

**Fix:**
*   **First Run Delay:** The first query takes longer because the model loads from disk. Wait a minute and try again.
*   **Pre-load the model:** Run this in a separate terminal:
    ```powershell
    ollama run llama3.2:3b ""
    ```
    This forces the model to load into memory *before* you start the backend.

---

### 4. "Model not found"
**Error:** `model 'llama3.2:3b' not found`

**Fix:**
Your teammate might not have the specific model installed.
*   Run:
    ```powershell
    ollama pull llama3.2:3b
    ollama pull gemma3:1b
    ```

---

## 🐛 Still stuck?

1. Send the output of `python verify_codebase.py` to the team.
2. Check `backend.log` if it exists.
