#!/usr/bin/env python3
"""
Build a Simple Chatbot
======================

Complete example of building a chatbot with StoneChain.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stonechain import Anthropic, Conversation, Tool, Agent


def simple_chatbot():
    """Basic chatbot with memory."""
    print("=" * 50)
    print("Simple Chatbot")
    print("=" * 50)
    print("Type 'quit' to exit.\n")
    
    llm = Anthropic()
    conv = Conversation(
        llm,
        system="You are a helpful, friendly assistant. Keep responses concise."
    )
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        response = conv.chat(user_input)
        print(f"Bot: {response}\n")


def chatbot_with_personality():
    """Chatbot with custom personality."""
    print("=" * 50)
    print("Pirate Chatbot")
    print("=" * 50)
    print("Type 'quit' to exit.\n")
    
    llm = Anthropic()
    conv = Conversation(
        llm,
        system="""You are a friendly pirate named Captain Claude.
        
Rules:
- Always speak like a pirate (use "arr", "matey", "ye", etc.)
- Be helpful but stay in character
- Keep responses under 3 sentences
- Occasionally mention your ship, the "SS StoneChain"
"""
    )
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Bot: Arr, fair winds to ye, matey! 🏴‍☠️")
            break
        
        if not user_input:
            continue
        
        response = conv.chat(user_input)
        print(f"Bot: {response}\n")


def chatbot_with_tools():
    """Chatbot that can use tools."""
    print("=" * 50)
    print("Chatbot with Tools")
    print("=" * 50)
    print("I can do math and tell time!")
    print("Type 'quit' to exit.\n")
    
    import datetime
    
    def calculator(expression: str) -> str:
        """Safe calculator."""
        try:
            allowed = set("0123456789+-*/.()")
            if all(c in allowed or c.isspace() for c in expression):
                return str(eval(expression))
            return "Invalid expression"
        except Exception as e:
            return f"Error: {e}"
    
    def get_time() -> str:
        """Get current time."""
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    tools = [
        Tool(
            "calculator",
            "Calculate math expressions like '2+2' or '15*23'",
            {"expression": {"type": "string", "description": "Math expression"}},
            calculator
        ),
        Tool(
            "get_time",
            "Get the current date and time",
            {},
            lambda: get_time()
        ),
    ]
    
    llm = Anthropic()
    agent = Agent(llm, tools, max_iterations=5)
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        result = agent.run(user_input)
        
        if result.get("answer"):
            print(f"Bot: {result['answer']}\n")
        else:
            print(f"Bot: Sorry, I couldn't figure that out.\n")


def chatbot_with_context():
    """Chatbot with document context (RAG-lite)."""
    print("=" * 50)
    print("Contextual Chatbot")
    print("=" * 50)
    print("Ask me about StoneChain!")
    print("Type 'quit' to exit.\n")
    
    context = """
StoneChain is a zero-dependency LLM framework created by Kent Stone in December 2025.

Key features:
- Zero external dependencies (pure Python stdlib)
- Single file distribution (~800 lines)
- Supports multiple providers: Anthropic, OpenAI, Groq, Mistral, DeepSeek, Ollama
- Features include: Chain, Router, Agent, RAG, Memory, Parallel execution

StoneChain was created as a response to LangChain's complexity. While LangChain has 
200+ dependencies and 100,000+ lines of code, StoneChain does the same in one file.

Installation is simple: just copy stonechain.py to your project. No pip install needed.

The philosophy is: "Built like a rock. Zero dependencies. Zero excuses."
"""
    
    llm = Anthropic()
    conv = Conversation(
        llm,
        system=f"""You are a helpful assistant that answers questions about StoneChain.

Here is information about StoneChain:
{context}

Rules:
- Answer based on the provided context
- If you don't know, say so
- Keep responses concise
- Be enthusiastic about StoneChain!
"""
    )
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        response = conv.chat(user_input)
        print(f"Bot: {response}\n")


def main():
    """Run chatbot examples."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY to run chatbot examples")
        return
    
    print("\nSelect a chatbot:")
    print("1. Simple chatbot")
    print("2. Pirate chatbot")
    print("3. Chatbot with tools")
    print("4. Contextual chatbot")
    print()
    
    choice = input("Choice (1-4): ").strip()
    print()
    
    if choice == "1":
        simple_chatbot()
    elif choice == "2":
        chatbot_with_personality()
    elif choice == "3":
        chatbot_with_tools()
    elif choice == "4":
        chatbot_with_context()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
