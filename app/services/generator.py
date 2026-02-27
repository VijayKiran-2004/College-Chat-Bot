import json
import requests

class Generator:
    """Handles prompt formulation and LLM response generation via Ollama"""
    
    def __init__(self, ollama_model, ollama_url):
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url

    def generate(self, query, docs, kb_context, links, language='en', stream=False, temperature=0.2):
        """Generate response using Ollama with retrieved context
        
        Args:
            query: User query
            docs: Retrieved documents
            kb_context: Formatted knowledge base string
            links: Extracted relevant links
            language: Response language ('en', 'hi', 'te')
            stream: If True, yields chunks. If False, returns complete response.
            temperature: LLM temperature (default 0.2 for factual)
        """
        
        # Build context from retrieved documents
        context = "\n\n".join([f"• {doc['contents'][:1000]}" for doc in docs[:5]])
        
        lang_instruction = ""
        if language == 'hi':
            lang_instruction = "IMPORTANT: Answer the student's question in HINDI (हिंदी). Transliterate technical terms if needed."
        elif language == 'te':
            lang_instruction = "IMPORTANT: Answer the student's question in TELUGU (తెలుగు). Transliterate technical terms if needed."
        else:
            lang_instruction = "Answer in English."

        prompt = f"""Hey! You're the friendly TKRCET College Buddy 😊 - think of yourself as a helpful senior student who knows everything about the college.
{lang_instruction}

YOUR PERSONALITY:
- Be warm, friendly, and conversational (like chatting with a friend)
- Use casual language but stay professional
- Show enthusiasm about TKRCET!
- Be understanding of typos and unclear questions

GUIDELINES:
- **Context is Key**: Always assume questions are about TKRCET. "What's the process?" = "TKRCET admission process"
- **Be Helpful**: If you don't have exact info, guide them to the right resource or office
- **Keep it Natural**: Avoid robotic responses - talk like a real person!

FORMATTING:
- Use **bold** for important names and numbers
- Use bullet points for lists
- **Be comprehensive yet concise**: Provide a complete answer but avoid fluff.
- If the question is complex, provide a step-by-step guide.

What I Know About TKRCET:
{kb_context}

Relevant Info:
{context}

Student's Question: {query}

Your Friendly Response:"""
        
        # Build Quick Links section (appears at top)
        quick_links_section = ""
        if links:
            quick_links_section = "📌 **Quick Links:**\n"
            for link in links:
                if isinstance(link, dict):
                    title = link['title']
                    url = link['url']
                elif 'tkrcet' in link.lower():
                    title = "TKRCET Official Page"
                    url = link
                else:
                    title = "Related Resource"
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
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": stream,
                    "options": {
                        "temperature": temperature,
                        "top_k": 40,
                        "top_p": 0.9,
                        "num_predict": 512,
                        "num_ctx": 2048
                    }
                },
                timeout=120,
                stream=stream
            )
            
            if stream:
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
            else:
                if response.status_code == 200:
                    answer = response.json().get('response', '').strip()
                    if answer:
                        return quick_links_section + answer + source_links_section
        except Exception as e:
            print(f"⚠ Ollama error: {e}")
            fallback = quick_links_section + f"Here's what I found:\n\n{context}" + source_links_section
            if stream:
                yield fallback
            else:
                return fallback
        
        if not stream:
            fallback = quick_links_section + f"Here's what I found:\n\n{context}" + source_links_section
            return fallback
