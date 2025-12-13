#!/usr/bin/env python3
"""
StoneChain Examples - Basic Usage
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stonechain import (
    Anthropic, OpenAI, Ollama, Groq,
    Message, Chain, Router, Agent, RAG, Conversation,
    Tool, Document, Parallel, complete
)


def example_simple():
    """Basic completion."""
    print("\n=== Simple Completion ===")
    llm = Anthropic()
    response = llm("What is the capital of France? One word.")
    print(f"Response: {response}")


def example_chain():
    """Chain example."""
    print("\n=== Chain ===")
    llm = Anthropic()
    
    chain = Chain(llm, system="You are helpful.")
    chain.add("analyze", "Analyze briefly: {input}", "analysis")
    chain.add("questions", "Generate 2 questions: {analysis}", "questions")
    
    result = chain.run(input="Machine Learning")
    print(f"Analysis: {result['outputs']['analysis'][:150]}...")
    print(f"Questions: {result['outputs']['questions'][:150]}...")


def example_agent():
    """Agent with tools."""
    print("\n=== Agent ===")
    
    def calculator(expression: str) -> str:
        try:
            return str(eval(expression))
        except:
            return "Error"
    
    tools = [
        Tool("calculator", "Do math", {"expression": {"type": "string"}}, calculator)
    ]
    
    agent = Agent(Anthropic(), tools, max_iterations=5)
    result = agent.run("What is 15 * 23 + 7?")
    print(f"Answer: {result['answer']}")


def example_rag():
    """RAG example."""
    print("\n=== RAG ===")
    
    rag = RAG(Anthropic())
    rag.add([
        Document("StoneChain was created by Kent Stone in December 2025."),
        Document("It has zero dependencies and uses only Python stdlib."),
    ])
    
    result = rag.query("Who created StoneChain?")
    print(f"Answer: {result['answer']}")


def example_conversation():
    """Conversation with memory."""
    print("\n=== Conversation ===")
    
    conv = Conversation(Anthropic(), system="You are a pirate. Be brief.")
    print(f"User: Hello!")
    print(f"Bot: {conv.chat('Hello!')}")
    print(f"User: What's your name?")
    print(f"Bot: {conv.chat('What is your name?')}")


def main():
    print("=" * 60)
    print("StoneChain Examples")
    print("=" * 60)
    
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\nSet ANTHROPIC_API_KEY to run examples")
        return
    
    example_simple()
    example_chain()
    example_agent()
    example_rag()
    example_conversation()
    
    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
