#!/usr/bin/env python3
"""
Terminal-based College Chatbot
Hybrid RAG-SQL System using Ollama and SQLite
"""

import sys
import os
from pathlib import Path

# Fix for WinError 1114 (DLL initialization failed)
# Must import sentence_transformers/torch BEFORE pandas/numpy
try:
    from sentence_transformers import SentenceTransformer  # noqa: F401
except ImportError:
    pass

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.query_router import QueryRouter  # noqa: E402


class TerminalChatbot:
    """Interactive terminal chatbot interface"""

    def __init__(self):
        """Initialize chatbot with Hybrid RAG-SQL system"""
        print("\n" + "=" * 70)
        print("COLLEGE CHATBOT - HYBRID RAG-SQL SYSTEM")
        print("=" * 70)
        print("Initializing chatbot system...")
        print()

        try:
            self.router = QueryRouter()
            self.running = True
            print("\n✓ Chatbot initialized successfully!")
            print("=" * 70)
        except Exception as e:
            print(f"\n✗ Error initializing chatbot: {str(e)}")
            print("Make sure:")
            print("  1. All data files are present")
            print("  2. Ollama service is running: ollama run llama3.2:3b")
            self.running = False

    def print_welcome(self):
        """Print welcome message"""
        print("\n" + "=" * 70)
        print("Welcome to TKRCET College Assistant Chatbot")
        print("=" * 70)
        print("\nYou can ask questions about:")
        print("  • College information (principal, timings, facilities)")
        print("  • Student data (CGPA, attendance, placements)")
        print("  • Departments and courses")
        print("  • Admissions and fees")
        print("\nType 'help' for available commands")
        print("Type 'exit' or 'quit' to exit")
        print("=" * 70 + "\n")

    def print_help(self):
        """Print help information"""
        print("\n" + "-" * 70)
        print("AVAILABLE COMMANDS:")
        print("-" * 70)
        print("  help          - Show this help message")
        print("  clear         - Clear screen")
        print("  status        - Show system status")
        print("  exit / quit   - Exit the chatbot")
        print("-" * 70)
        print("\nJust type your question and press Enter to get an answer!")
        print("-" * 70 + "\n")

    def print_status(self):
        """Print system status"""
        print("\n" + "-" * 70)
        print("SYSTEM STATUS")
        print("-" * 70)
        try:
            # Access internal components for status
            rag = self.router.rag_system
            sql = self.router.sql_system

            num_docs = len(rag.documents)
            print(f"✓ Knowledge Base: {num_docs} documents loaded")
            print("✓ Embedding Model: all-MiniLM-L6-v2")
            print(f"✓ LLM Model: {rag.ollama_model} (via Ollama)")
            print(f"✓ SQL Database: Connected to {sql.db_path}")
            print("✓ Retrieval: Hybrid (FAISS + BM25) + SQL")
            print("-" * 70 + "\n")
        except Exception as e:
            print(f"✗ Error getting status: {str(e)}\n")

    def run(self):
        """Run the interactive chatbot"""
        if not self.running:
            print("\n✗ Chatbot could not be initialized. Exiting.")
            return

        self.print_welcome()

        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.lower() in ["exit", "quit"]:
                    print("\n" + "=" * 70)
                    print("Thank you for using TKRCET College Assistant!")
                    print("=" * 70 + "\n")
                    self.router.close()
                    break

                elif user_input.lower() == "help":
                    self.print_help()

                elif user_input.lower() == "clear":
                    os.system("cls" if os.name == "nt" else "clear")
                    self.print_welcome()

                elif user_input.lower() == "status":
                    self.print_status()

                else:
                    # Process query through Hybrid system
                    print("\nAssistant: ", end="", flush=True)

                    try:
                        # Direct call to router logic
                        # Print newline to avoid router log overlap
                        # Actually router returns a string, so we just print it.
                        answer = self.router(user_input)
                        print(answer)
                    except Exception as e:
                        print(f"[Error processing query: {str(e)}]")

                    print()

            except KeyboardInterrupt:
                print("\n\n" + "=" * 70)
                print("Chatbot interrupted. Goodbye!")
                print("=" * 70 + "\n")
                self.router.close()
                break
            except Exception as e:
                print(f"\n✗ Error: {str(e)}\n")


def main():
    """Main entry point"""
    chatbot = TerminalChatbot()
    chatbot.run()


if __name__ == "__main__":
    main()
