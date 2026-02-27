"""
Deep Reasoning Chain
Orchestrates complex queries using LangChain Agents, RAG, and SQL.
"""

from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.chat_models import ChatOllama
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from app.services.ultra_rag import UltraRAGSystem
from app.services.sql_system import SQLSystem
import sys
import io

# Fix Windows encoding handled in ultra_rag.py

class DeepReasoningChain:
    """
    Intelligent Agent that can reason about which tool to use.
    Tools:
    1. College Knowledge (RAG) - For general info, syllabus, faculty, etc.
    2. Student Database (SQL) - For student counts, placements, marks, etc.
    """
    
    def __init__(self, rag_system=None, sql_system=None):
        print("\n" + "="*50)
        print("Initializing Deep Reasoning Chain...")
        
        # 1. Initialize Systems (Use injected or create new)
        self.rag_system = rag_system if rag_system else UltraRAGSystem()
        self.sql_system = sql_system if sql_system else SQLSystem()
        
        # 2. Define Tools
        tools = [
            Tool(
                name="CollegeKnowledge",
                func=self._rag_wrapper,
                description="Use this for general questions about the college, syllabus, faculty, history, timings, rules, or static information. Input should be the full question."
            ),
            Tool(
                name="StudentDatabase",
                func=self._sql_wrapper,
                description="Use this ONLY for queries about student data, counts, placements, companies, specific student details, or existing database stats. Input should be the full question."
            )
        ]
        
        # 3. Initialize LLM (Llama 3.2 via Ollama)
        llm = ChatOllama(
            model="llama3.2:3b",
            temperature=0, # Zero temp for precise tool use
            base_url="http://localhost:11434"
        )
        
        # 4. Create ReAct Agent
        # Explicit instructions for the agent
        template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

        prompt = PromptTemplate.from_template(template)
        
        agent = create_react_agent(llm, tools, prompt)
        
        self.agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True, # Show thinking process in console
            handle_parsing_errors=True,
            max_iterations=5
        )
        print("✓ Deep Reasoning Agent Ready")

    def _rag_wrapper(self, query: str) -> str:
        """Wrapper for RAG retrieval"""
        print(f"  [Agent] Calling RAG for: {query}")
        return self.rag_system(query)

    def _sql_wrapper(self, query: str) -> str:
        """Wrapper for SQL query"""
        print(f"  [Agent] Calling SQL for: {query}")
        # Note: SQLSystem usually returns a DataFrame string representation
        return self.sql_system.query_students(query)

    def run(self, query: str) -> str:
        """Run the reasoning chain"""
        try:
            print(f"\n[DeepChain] Processing: {query}")
            result = self.agent_executor.invoke({"input": query})
            return result['output']
        except Exception as e:
            print(f"⚠ Chain Error: {e}")
            # Fallback to simple RAG if agent fails
            return self.rag_system(query)
