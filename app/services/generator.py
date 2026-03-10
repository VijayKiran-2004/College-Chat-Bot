import json
import re
import requests
from pathlib import Path


class Generator:
    """Handles prompt formulation and LLM response generation via Ollama"""

    # Signals that a query needs a detailed, step-by-step answer
    COMPLEX_PATTERNS = re.compile(
        r"\b(how to|procedure|steps|process|apply for|guide|explain|describe|"
        r"difference|compare|what are the|list all|tell me about|"
        r"|what is the procedure|can you explain|give me details)\b",
        re.IGNORECASE,
    )

    def __init__(self, ollama_model, ollama_url):
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url

        # Load Soul from JSON
        self.soul = self._load_soul()
        print(f"  ✓ Soul loaded (v{self.soul.get('_version', 'unknown')})")

    def _load_soul(self) -> dict:
        """Load the bot personality/soul from app/config/soul.json"""
        soul_path = Path(__file__).resolve().parent.parent / "config" / "soul.json"
        try:
            with open(soul_path, "r", encoding="utf-8") as f:
                soul = json.load(f)
            return soul
        except FileNotFoundError:
            print(f"⚠ Soul file not found at {soul_path}. Using built-in defaults.")
            return self._default_soul()
        except json.JSONDecodeError as e:
            print(f"⚠ Soul file has invalid JSON ({e}). Using built-in defaults.")
            return self._default_soul()

    def _default_soul(self) -> dict:
        """Hardcoded fallback soul in case soul.json is missing or corrupt"""
        return {
            "_version": "fallback",
            "compact": (
                "You are College Buddy, the official TKRCET assistant."
                " Answer ONLY from the provided context. Never guess. "
                "If the context doesn't have the answer, "
                "say you don't have that information."
            ),
            "complex_extension": (
                "Use numbered lists for procedures. Bold key terms. Be thorough.",
            ),
            "greeting": (
                "You are College Buddy, the official TKRCET assistant."
                " Respond with a short, warm greeting and ask how you can help.",
            ),
            "fallback_no_info": (
                "I don't have specific information on that right now."
                " Please reach out to the college office or visit tkrcet.ac.in.",
            ),
            "fallback_off_topic": (
                "I'm here to help with TKRCET-related queries only!"
                " Is there something about the college I can assist you with?"
            ),
            "fallback_error": (
                "I'm having a little trouble right now."
                " Please try again in a moment, or visit tkrcet.ac.in for help."
            ),
        }

    def _is_complex_query(self, query: str) -> bool:
        """Quick regex-based complexity check — no model call needed."""
        return bool(self.COMPLEX_PATTERNS.search(query))

    def generate(
        self,
        query,
        docs,
        kb_context,
        links,
        kb_fact=None,
        language="en",
        stream=False,
        temperature=0.2,
        is_greeting=False,
    ):
        """Generate response using Ollama with retrieved context

        Args:
            query: User query
            docs: Retrieved documents
            kb_context: Formatted knowledge base string
            links: Extracted relevant links
            kb_fact: Raw fact from Knowledge Base (if any)
            language: Response language ('en', 'hi', 'te')
            stream: If True, yields chunks. If False, returns complete response.
            temperature: LLM temperature (default 0.2 for factual)
            is_greeting: If True, uses a lite prompt for greetings
        """

        # ── Greeting fast-track ──────────────────────────────────
        if is_greeting:
            greeting_soul = self.soul.get("greeting", self._default_soul()["greeting"])
            lite_prompt = f'{greeting_soul}\n\nThe user said: "{query}"'
            try:
                if stream:
                    return self._generate_stream(lite_prompt, "", "", temperature)
                else:
                    return self._generate_sync(lite_prompt, "", "", temperature)
            except Exception as e:
                print(f"⚠ Greeting Fast-track error: {e}")
                fallback = (
                    "Hello! I'm your TKRCET College Buddy. How can I help you today?"
                )
                return self._yield_fallback(fallback) if stream else fallback

        # ── Build context pieces ─────────────────────────────────
        # Prioritize raw KB Fast Track facts if available
        found_fact_section = ""
        if kb_fact:
            found_fact_section = f"\nCORE FACT (Priority Source):\n{kb_fact}\n"

        # Build context from retrieved documents
        # (Reduce length to 500 chars to save prefill time)
        context = "\n\n".join([f"• {doc['contents'][:500]}" for doc in docs[:3]])

        lang_instruction = "Answer in English."

        # ── Dual-Prompt Strategy (Soul-Powered) ──────────────────
        # Complex queries get compact + complex_extension soul.
        # Simple queries get compact soul only.
        soul_core = self.soul.get("compact", self._default_soul()["compact"])
        is_complex = self._is_complex_query(query)

        if is_complex:
            print("  [Generator] Complex query detected — using detailed prompt.")
            soul_extension = self.soul.get(
                "complex_extension", self._default_soul()["complex_extension"]
            )
            num_predict = 500
            prompt = f"""{soul_core}

{soul_extension}
{lang_instruction}

Context:
{kb_context}
{found_fact_section}
{context}

Question: {query}
Detailed Answer:"""
        else:
            print("  [Generator] Simple query — using compact prompt.")
            num_predict = 300
            prompt = f"""{soul_core}
{lang_instruction}

Context:
{kb_context}
{found_fact_section}
{context}

Question: {query}
Answer:"""

        # Build Quick Links section (appears at top)
        quick_links_section = ""
        if links:
            quick_links_section = "📌 **Quick Links:**\n"
            seen_urls = set()
            for link in links:
                url = link.get("url") if isinstance(link, dict) else link
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                if isinstance(link, dict):
                    title = link.get("title", "Official Link")
                elif "tkrcet" in link.lower():
                    title = "TKRCET Official Page"
                else:
                    title = "View Resource"
                quick_links_section += f"• [{title}]({url})\n"
            quick_links_section += "\n"

        try:
            if stream:
                return self._generate_stream(
                    prompt, quick_links_section, context, temperature, num_predict
                )
            else:
                return self._generate_sync(
                    prompt, quick_links_section, context, temperature, num_predict
                )
        except requests.exceptions.ConnectionError:
            error_msg = (
                "✗ Connection Error: Ollama is not running at 127.0.0.1:11434."
                " Please start it with 'ollama serve'."
            )
            print(f"⚠ {error_msg}")
            return quick_links_section + f"⚠ {error_msg}\n\nRaw Context:\n{context}"
        except Exception as e:
            print(f"⚠ Ollama error: {e}")
            fallback_msg = self.soul.get(
                "fallback_error", "I'm having trouble right now. Please try again."
            )
            fallback = quick_links_section + f"⚠ {fallback_msg}"
            return self._yield_fallback(fallback) if stream else fallback

    def _generate_sync(
        self, prompt, quick_links_section, context, temperature, num_predict=150
    ):
        """Internal sync generation"""
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": float(temperature),
                        "num_ctx": 2048,
                        "num_predict": num_predict,
                    },
                },
                timeout=120,
            )
            if response.status_code == 200:
                answer = response.json().get("response", "").strip()
                if answer:
                    return quick_links_section + answer
            elif response.status_code == 404:
                return (
                    quick_links_section
                    + f"⚠ Model Not Found: '{self.ollama_model}' is not installed."
                    f" Please run 'ollama pull {self.ollama_model}'."
                )

            # Extract detailed error from Ollama if available
            err_detail = ""
            try:
                err_json = response.json()
                if "error" in err_json:
                    err_detail = f" Detail: {err_json['error']}"
            except Exception:
                err_detail = f" Detail: {response.text[:100]}"

            return (
                quick_links_section + f"⚠ Ollama returned an error"
                f"(Status {response.status_code}).{err_detail}"
            )
        except requests.exceptions.ConnectionError:
            return (
                quick_links_section + "⚠ Connection Error: Could not reach Ollama."
                " Is 'ollama serve' running?"
            )
        except Exception as e:
            return (
                quick_links_section + f"⚠ I couldn't generate a response. "
                f"Error: {str(e)}\n\nContext: {context[:200]}..."
            )

    def _generate_stream(
        self, prompt, quick_links_section, context, temperature, num_predict=150
    ):
        """Internal stream generation"""
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "temperature": float(temperature),
                        "num_ctx": 2048,
                        "num_predict": num_predict,
                    },
                },
                timeout=120,
                stream=True,
            )

            if response.status_code == 404:
                yield "⚠ Model Not Found:"
                yield f" The model '{self.ollama_model}' is not installed."
                yield f" Please run 'ollama pull {self.ollama_model}'."
                return
            elif response.status_code != 200:
                err_detail = ""
                try:
                    err_json = response.json()
                    if "error" in err_json:
                        err_detail = f" Detail: {err_json['error']}"
                except Exception:
                    err_detail = f" Detail: {response.text[:100]}"
                yield f"⚠ Ollama error (Status {response.status_code}).{err_detail}"
                return

            if quick_links_section:
                yield quick_links_section

            for line in response.iter_lines():
                if line:
                    try:
                        chunk_data = json.loads(line)
                        if "response" in chunk_data:
                            yield chunk_data["response"]
                        if chunk_data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
        except requests.exceptions.ConnectionError:
            yield "⚠ Connection Error: Could not reach Ollama."
            yield " Is 'ollama serve' running?"
        except Exception as e:
            yield f"⚠ Streaming error: {str(e)}"

    def _yield_fallback(self, fallback):
        yield fallback
