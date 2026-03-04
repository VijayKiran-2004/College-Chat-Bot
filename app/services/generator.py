import json
import re

import requests


class Generator:
    """Handles prompt formulation and LLM response generation via Ollama."""

    COMPLEX_PATTERNS = re.compile(
        (
            r"\b(how to|procedure|steps|process|apply for|guide|"
            r"explain|describe|difference|compare|what are the|"
            r"list all|tell me about|what is the procedure|"
            r"can you explain|give me details)\b"
        ),
        re.IGNORECASE,
    )

    def __init__(self, ollama_model, ollama_url):
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url

    def _is_complex_query(self, query: str) -> bool:
        """Quick regex-based complexity check."""
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
        """Generate response using Ollama with retrieved context."""

        if is_greeting:
            lite_prompt = (
                "You are the official TKRCET College Buddy. "
                f"The user said '{query}'. "
                "Respond with a short and friendly greeting."
            )

            try:
                if stream:
                    return self._generate_stream(
                        lite_prompt, "", "", "", temperature
                    )
                return self._generate_sync(
                    lite_prompt, "", "", "", temperature
                )
            except Exception:
                fallback = (
                    "Hello! I'm your TKRCET College Buddy. "
                    "How can I help you today?"
                )
                return self._yield_fallback(fallback) if stream else fallback

        found_fact_section = ""
        if kb_fact:
            found_fact_section = (
                "\nCORE FACT (Priority Source):\n"
                f"{kb_fact}\n"
            )

        context = "\n\n".join(
            [f"• {doc['contents'][:500]}" for doc in docs[:3]]
        )

        if language == "hi":
            lang_instruction = "Answer in HINDI (हिंदी)."
        elif language == "te":
            lang_instruction = "Answer in TELUGU (తెలుగు)."
        else:
            lang_instruction = "Answer in English."

        is_complex = self._is_complex_query(query)

        if is_complex:
            num_predict = 300
            prompt = (
                f"You are the TKRCET College Buddy. "
                f"{lang_instruction}\n\n"
                "Use ONLY the provided context.\n\n"
                f"{kb_context}\n"
                f"{found_fact_section}"
                f"{context}\n\n"
                f"Question: {query}\n"
                "Detailed Answer:"
            )
        else:
            num_predict = 150
            prompt = (
                f"You are the TKRCET College Buddy. "
                f"{lang_instruction}\n\n"
                "Use ONLY the provided context.\n\n"
                f"{kb_context}\n"
                f"{found_fact_section}"
                f"{context}\n\n"
                f"Question: {query}\n"
                "Answer:"
            )

        quick_links_section = ""
        if links:
            quick_links_section = "📌 **Quick Links:**\n"
            for link in links:
                if isinstance(link, dict):
                    title = link.get("title", "Official Link")
                    url = link.get("url", "#")
                else:
                    title = "View Resource"
                    url = link
                quick_links_section += f"• [{title}]({url})\n"
            quick_links_section += "\n"

        source_links_section = ""
        if links:
            source_links_section = "\n\n📚 **Source Links:**\n"
            for link in links:
                if isinstance(link, dict):
                    source_links_section += (
                        f"• [{link['title']}]({link['url']})\n"
                    )
                else:
                    source_links_section += f"• {link}\n"

        try:
            if stream:
                return self._generate_stream(
                    prompt,
                    quick_links_section,
                    source_links_section,
                    context,
                    temperature,
                    num_predict,
                )
            return self._generate_sync(
                prompt,
                quick_links_section,
                source_links_section,
                context,
                temperature,
                num_predict,
            )
        except Exception:
            fallback = (
                quick_links_section
                + "Here's what I found:\n\n"
                + context
                + source_links_section
            )
            return self._yield_fallback(fallback) if stream else fallback

    def _generate_sync(
        self,
        prompt,
        quick_links_section,
        source_links_section,
        context,
        temperature,
        num_predict=150,
    ):
        response = requests.post(
            self.ollama_url,
            json={
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_ctx": 2048,
                    "num_predict": num_predict,
                },
            },
            timeout=120,
        )

        if response.status_code == 200:
            answer = response.json().get("response", "").strip()
            if answer:
                return quick_links_section + answer + source_links_section

        return (
            quick_links_section
            + "Here's what I found:\n\n"
            + context
            + source_links_section
        )

    def _generate_stream(
        self,
        prompt,
        quick_links_section,
        source_links_section,
        context,
        temperature,
        num_predict=150,
    ):
        response = requests.post(
            self.ollama_url,
            json={
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "num_ctx": 2048,
                    "num_predict": num_predict,
                },
            },
            timeout=120,
            stream=True,
        )

        if quick_links_section:
            yield quick_links_section

        for line in response.iter_lines():
            if line:
                try:
                    chunk_data = json.loads(line)
                    if "response" in chunk_data:
                        yield chunk_data["response"]
                    if chunk_data.get("done", False):
                        if source_links_section:
                            yield source_links_section
                        break
                except json.JSONDecodeError:
                    continue

    def _yield_fallback(self, fallback):
        yield fallback
