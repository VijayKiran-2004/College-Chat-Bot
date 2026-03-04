"""
Deep Reasoning Chain
Orchestrates complex queries using LangChain Agents, RAG, and SQL.
"""

from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool

from app.services.sql_system import SQLSystem
from app.services.ultra_rag import UltraRAGSystem


class DeepReasoningChain:
    """
    Intelligent Agent that can reason about which tool to use.

    Tools:
    1. College Knowledge (RAG)
    2. Student Database (SQL)
    """

    def __init__(self, rag_system=None, sql_system=None):
        print("\n" + "=" * 50)
        print("Initializing Deep Reasoning Chain...")

        self.rag_system = rag_system if rag_system else UltraRAGSystem()
        self.sql_system = sql_system if sql_system else SQLSystem()

        tools = [
            Tool(
                name="CollegeKnowledge",
                func=self._rag_wrapper,
                description=(
                    "Use this for general questions about the college, "
                    "syllabus, faculty, history, timings, rules, or "
                    "static information. Input should be the full question."
                ),
            ),
            Tool(
                name="StudentDatabase",
                func=self._sql_wrapper,
                description=(
                    "Use this ONLY for queries about student data, counts, "
                    "placements, companies, specific student details, or "
                    "existing database stats. "
                    "Input should be the full question."
                ),
            ),
        ]

        llm = ChatOllama(
            model="llama3.2:3b",
            temperature=0,  # Zero temp for precise tool use
            base_url="http://localhost:11434",
        )

        template = """Answer the following questions as best you can...
        """

        prompt = PromptTemplate.from_template(template)
        agent = create_react_agent(llm, tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
        )

        print("✓ Deep Reasoning Agent Ready")
