"""
Enhanced Conversational Agent with Memory
This agent uses LangChain with Claude 3 and includes:
- Conversation memory using ChromaDB
- Web search capability
- Structured output handling
- Multiple tool support
"""

import os
from typing import List, Dict
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import Tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
import chromadb
from chromadb.config import Settings

# Load environment variables
load_dotenv()

class ConversationalAgent:
    def __init__(self):
        # Initialize ChromaDB
        self.chroma_client = chromadb.Client(Settings(
            persist_directory="vector_store",
            is_persistent=True
        ))
        
        # Create or get the collection for conversation history
        self.collection = self.chroma_client.get_or_create_collection(
            name="conversation_history"
        )
        
        # Initialize memory
        self.memory = MemorySaver()
        
        # Initialize LLM
        self.model = ChatAnthropic(
            model_name="claude-3-sonnet-20240229",
            temperature=0.7
        )
        
        # Initialize tools
        self.search = TavilySearchResults(max_results=3)
        self.tools = [
            self.search,
            Tool(
                name="memory_search",
                description="Search through past conversations",
                func=self.search_memory
            )
        ]
        
        # Create the agent
        self.agent = create_react_agent(
            self.model,
            self.tools,
            checkpointer=self.memory
        )

    def search_memory(self, query: str) -> str:
        """Search through conversation history."""
        results = self.collection.query(
            query_texts=[query],
            n_results=2
        )
        if results and results['documents']:
            return "Related past conversations:\n" + "\n".join(results['documents'][0])
        return "No relevant past conversations found."

    def save_to_memory(self, human_msg: str, ai_response: str):
        """Save conversation to ChromaDB."""
        self.collection.add(
            documents=[f"Human: {human_msg}\nAI: {ai_response}"],
            ids=[f"conv_{len(self.collection.get()['ids']) + 1}"]
        )

    def chat(self, message: str, thread_id: str = "default") -> str:
        """Process a message and return the response."""
        config = {"configurable": {"thread_id": thread_id}}
        response = ""
        
        # Process message through agent
        for chunk in self.agent.stream(
            {"messages": [HumanMessage(content=message)]},
            config
        ):
            if isinstance(chunk, AIMessage):
                response += chunk.content
            print(chunk)
        
        # Save conversation to memory
        self.save_to_memory(message, response)
        return response

def main():
    # Initialize agent
    agent = ConversationalAgent()
    
    print("🤖 Conversational Agent initialized! (Type 'exit' to quit)")
    print("---------------------------------------------------")
    
    thread_id = "chat_" + os.urandom(4).hex()
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == 'exit':
            break
            
        try:
            response = agent.chat(user_input, thread_id)
            print("\nAI:", response)
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again.")

if __name__ == "__main__":
    main()
