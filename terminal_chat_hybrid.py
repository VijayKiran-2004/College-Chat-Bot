"""
Terminal Chat with Hybrid RAG-SQL System
Intelligent chatbot that routes queries to RAG or SQL based on intent
"""
import sys
from app.services.query_router import QueryRouter

def print_header():
    print("\n" + "="*70)
    print("COLLEGE CHATBOT - HYBRID RAG-SQL SYSTEM")
    print("="*70)
    print("Initializing chatbot system...")
    print()

def print_welcome():
    print("="*70)
    print("Welcome to TKRCET College Assistant Chatbot")
    print("="*70)
    print()
    print("You can ask questions about:")
    print("  • College information (principal, timings, facilities)")
    print("  • Student data (CGPA, attendance, placements)")
    print("  • Departments and courses")
    print("  • Admissions and fees")
    print()
    print("Type 'help' for available commands")
    print("Type 'exit' or 'quit' to exit")
    print("="*70)
    print()

def print_help():
    print("\nAvailable commands:")
    print("  help    - Show this help message")
    print("  clear   - Clear conversation history")
    print("  exit    - Exit the chatbot")
    print("  quit    - Exit the chatbot")
    print()
    print("Example queries:")
    print("  General: 'Who is the principal?', 'What are college timings?'")
    print("  Student: 'List CSE students', 'Show students with CGPA > 8.5'")
    print()

def main():
    print_header()
    
    # Initialize query router
    try:
        router = QueryRouter()
    except Exception as e:
        print(f"Error initializing chatbot: {e}")
        sys.exit(1)
    
    print_welcome()
    
    # Main chat loop
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['exit', 'quit']:
                print("\nThank you for using TKRCET College Assistant. Goodbye!")
                router.close()
                break
            
            elif user_input.lower() == 'help':
                print_help()
                continue
            
            elif user_input.lower() == 'clear':
                print("\nConversation history cleared.")
                continue
            
            # Process query
            print()
            response = router(user_input)
            print(f"Assistant: {response}")
            print()
        
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            router.close()
            break
        
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.\n")

if __name__ == "__main__":
    main()
