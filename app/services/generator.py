import json
import re
import requests

class Generator:
    """Handles prompt formulation and LLM response generation via Ollama"""
    
    # Signals that a query needs a detailed, step-by-step answer
    COMPLEX_PATTERNS = re.compile(
        r'\b(how to|procedure|steps|process|apply for|guide|explain|describe|'  
        r'difference|compare|what are the|list all|tell me about|what is the procedure|'  
        r'can you explain|give me details)\b',
        re.IGNORECASE
    )

    def __init__(self, ollama_model, ollama_url):
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url

    def _is_complex_query(self, query: str) -> bool:
        """Quick regex-based complexity check — no model call needed."""
        return bool(self.COMPLEX_PATTERNS.search(query))

    def generate(self, query, docs, kb_context, links, kb_fact=None, language='en', stream=False, temperature=0.2, is_greeting=False):
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
        
        if is_greeting:
            lite_prompt = f"You are the official TKRCET College Buddy. The user just said '{query}'. " \
                          f"Respond with a short, professional, and very friendly greeting and ask how you can help them today with college-related queries. " \
                          f"Keep it under 2 sentences and very welcoming. Answer in English unless they spoke to you in another language."
            try:
                if stream:
                    return self._generate_stream(lite_prompt, "", "", "", temperature)
                else:
                    return self._generate_sync(lite_prompt, "", "", "", temperature)
            except Exception as e:
                print(f"⚠ Greeting Fast-track error: {e}")
                return self._yield_fallback("Hello! I'm your TKRCET College Buddy. How can I help you today?") if stream else "Hello! I'm your TKRCET College Buddy. How can I help you today?"

        # Prioritize raw KB Fast Track facts if available
        found_fact_section = ""
        if kb_fact:
            found_fact_section = f"\nCORE FACT (Priority Source):\n{kb_fact}\n"

        # Build context from retrieved documents (Reduce length to 500 chars to save prefill time)
        context = "\n\n".join([f"• {doc['contents'][:500]}" for doc in docs[:3]])
        
        lang_instruction = ""
        if language == 'hi':
            lang_instruction = "Answer in HINDI (हिंदी)."
        elif language == 'te':
            lang_instruction = "Answer in TELUGU (తెలుగు)."
        else:
            lang_instruction = "Answer in English."

        # --- Dual-Prompt Strategy ---
        # Complex queries (procedures, lists) get a richer prompt for accuracy.
        # Simple queries keep the compact prompt for maximum speed.
        is_complex = self._is_complex_query(query)

        if is_complex:
            print(f"  [Generator] Complex query detected — using detailed prompt.")
            num_predict = 200  # Capped from 300 — saves ~25s on low-end hardware
            prompt = f"""You are the TKRCET College Buddy. {lang_instruction}
Use ONLY the provided context. Be thorough and structured for this detailed question about TKRCET.
If steps are involved, use a numbered list. Use **bold** for key terms.

Context:
{kb_context}
{found_fact_section}
{context}

Question: {query}
Detailed Answer:"""
        else:
            print(f"  [Generator] Simple query — using compact prompt.")
            num_predict = 150  # Short, fast answer
            prompt = f"""You are the TKRCET College Buddy. {lang_instruction}
Rules: Use ONLY the provided context. Be concise. Assume all questions are about TKRCET.

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
            for link in links:
                if isinstance(link, dict):
                    title = link.get('title', 'Official Link')
                    url = link.get('url', '#')
                elif 'tkrcet' in link.lower():
                    title = "TKRCET Official Page"
                    url = link
                else:
                    title = "View Resource"
                    url = link
                quick_links_section += f"• [{title}]({url})\n"
            quick_links_section += "\n"
        
        # Build source links footer
        source_links_section = ""
        if links:
            source_links_section = "\n\n📚 **Source Links:**\n"
            for link in links:
                if isinstance(link, dict):
                    source_links_section += f"• [{link['title']}]({link['url']})\n"
                else:
                    source_links_section += f"• {link}\n"
        
        try:
            if stream:
                return self._generate_stream(prompt, quick_links_section, source_links_section, context, temperature, num_predict)
            else:
                return self._generate_sync(prompt, quick_links_section, source_links_section, context, temperature, num_predict)
        except Exception as e:
            print(f"⚠ Ollama error: {e}")
            fallback = quick_links_section + f"⚠ I couldn't generate a full response right now (Ollama may be busy). Here's the raw context I found:\n\n{context}" + source_links_section
            return self._yield_fallback(fallback) if stream else fallback

    def _generate_sync(self, prompt, quick_links_section, source_links_section, context, temperature, num_predict=150):
        """Internal sync generation"""
        response = requests.post(
            self.ollama_url,
            json={
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature, 
                    "num_ctx": 2048,
                    "num_predict": num_predict
                }
            },
            timeout=120
        )
        if response.status_code == 200:
            answer = response.json().get('response', '').strip()
            if answer:
                return quick_links_section + answer + source_links_section
        
        return quick_links_section + f"⚠ I couldn't generate a full response right now (Ollama may be busy). Here's the raw context I found:\n\n{context}" + source_links_section

    def _generate_stream(self, prompt, quick_links_section, source_links_section, context, temperature, num_predict=150):
        """Internal stream generation"""
        response = requests.post(
            self.ollama_url,
            json={
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": temperature, 
                    "num_ctx": 2048,
                    "num_predict": num_predict
                }
            },
            timeout=120,
            stream=True
        )
        
        if quick_links_section:
            yield quick_links_section
        
        for line in response.iter_lines():
            if line:
                try:
                    chunk_data = json.loads(line)
                    if 'response' in chunk_data:
                        yield chunk_data['response']
                    if chunk_data.get('done', False):
                        if source_links_section:
                            yield source_links_section
                        break
                except json.JSONDecodeError:
                    continue

    def _yield_fallback(self, fallback):
        yield fallback
