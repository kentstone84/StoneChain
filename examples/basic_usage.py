"""
SimpleChain Examples
====================

Quick examples to get you started.
"""

import os
import asyncio

# Make sure we can import simplechain
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simplechain import (
    Anthropic, OpenAI, Ollama, Groq,
    Message, Chain, Router, Agent, RAG, Conversation,
    Tool, Document, Parallel,
    complete
)


def example_simple_completion():
    """Basic completion example."""
    print("\n=== Simple Completion ===")
    
    # Using environment variable
    llm = Anthropic()
    
    # Shorthand
    response = llm("What is the capital of France? One word.")
    print(f"Response: {response}")


def example_with_messages():
    """Using Message objects for more control."""
    print("\n=== With Messages ===")
    
    llm = Anthropic()
    
    response = llm.complete([
        Message.system("You are a helpful geography teacher."),
        Message.user("What is the capital of Japan?"),
    ])
    
    print(f"Response: {response.content}")
    print(f"Tokens: {response.total_tokens}")
    print(f"Latency: {response.latency_ms:.0f}ms")


def example_chain():
    """Chain example - sequential LLM calls."""
    print("\n=== Chain ===")
    
    llm = Anthropic()
    
    chain = Chain(llm, system="You are a helpful assistant.")
    chain.add("analyze", "Analyze this topic briefly: {input}", "analysis")
    chain.add("questions", "Generate 2 questions about: {analysis}", "questions")
    
    result = chain.run(input="Machine Learning")
    
    print(f"Analysis: {result['outputs']['analysis'][:200]}...")
    print(f"Questions: {result['outputs']['questions']}")
    print(f"Total tokens: {result['total_tokens']}")


def example_router():
    """Router example - route to different handlers."""
    print("\n=== Router ===")
    
    llm = Anthropic()
    
    # Create specialized chains
    code_chain = Chain(llm, system="You are a coding expert. Be concise.")
    code_chain.add("code", "Write code for: {input}", "result")
    
    math_chain = Chain(llm, system="You are a math expert. Be concise.")
    math_chain.add("solve", "Solve: {input}", "result")
    
    # Router
    router = Router(default=llm)
    router.add("code", lambda x: "code" in x.lower() or "python" in x.lower(), code_chain)
    router.add("math", lambda x: any(c in x for c in "+-*/="), math_chain)
    
    # Test routing
    queries = [
        "Write python code to add two numbers",
        "What is 15 + 27?",
        "Tell me about the weather",
    ]
    
    for query in queries:
        result = router.route(query)
        route = result.get("routed_to", "unknown")
        output = result.get("outputs", {}).get("result", result.get("response", ""))
        print(f"Query: {query[:40]}...")
        print(f"  Routed to: {route}")
        print(f"  Output: {output[:100]}...")
        print()


def example_agent():
    """Agent example - tool use."""
    print("\n=== Agent ===")
    
    llm = Anthropic()
    
    # Define tools
    def calculator(expression: str) -> str:
        """Safe calculator."""
        try:
            # Only allow numbers and basic operators
            allowed = set("0123456789+-*/.()")
            if not all(c in allowed or c.isspace() for c in expression):
                return "Error: Invalid characters"
            return str(eval(expression))
        except Exception as e:
            return f"Error: {e}"
    
    tools = [
        Tool(
            name="calculator",
            description="Calculate math expressions. Input: expression string",
            parameters={"expression": {"type": "string", "description": "Math expression"}},
            function=calculator
        )
    ]
    
    agent = Agent(llm, tools, max_iterations=5)
    
    result = agent.run("What is 15 * 23 + 7?")
    print(f"Answer: {result['answer']}")
    print(f"Iterations: {len(result['iterations'])}")


def example_rag():
    """RAG example - document Q&A."""
    print("\n=== RAG ===")
    
    llm = Anthropic()
    rag = RAG(llm, top_k=2)
    
    # Add documents
    rag.add([
        Document("SimpleChain was created by Kent Stone in December 2025."),
        Document("SimpleChain has zero dependencies and uses only Python stdlib."),
        Document("SimpleChain supports Anthropic, OpenAI, Groq, Mistral, DeepSeek, and Ollama."),
        Document("The entire codebase is about 800 lines of Python."),
    ])
    
    # Query
    questions = [
        "Who created SimpleChain?",
        "What providers does SimpleChain support?",
    ]
    
    for q in questions:
        result = rag.query(q)
        print(f"Q: {q}")
        print(f"A: {result['answer']}")
        print()


def example_conversation():
    """Conversation example - with memory."""
    print("\n=== Conversation ===")
    
    llm = Anthropic()
    conv = Conversation(llm, system="You are a friendly pirate. Keep responses short.")
    
    exchanges = [
        "Hello!",
        "What's your name?",
        "What did I just ask you?",  # Tests memory
    ]
    
    for msg in exchanges:
        response = conv.chat(msg)
        print(f"User: {msg}")
        print(f"Bot: {response}")
        print()


def example_parallel():
    """Parallel execution example."""
    print("\n=== Parallel ===")
    
    llm = Anthropic()
    
    # Run multiple queries in parallel
    queries = [
        "What is 2+2?",
        "Capital of France?",
        "Largest planet?",
    ]
    
    tasks = [(llm, q) for q in queries]
    results = Parallel.run(tasks, max_workers=3)
    
    for q, r in zip(queries, results):
        print(f"Q: {q} -> A: {r.content[:50]}")


async def example_async():
    """Async example."""
    print("\n=== Async ===")
    
    llm = Anthropic()
    
    # Async completion
    response = await llm.acomplete([Message.user("What is 3+3?")])
    print(f"Async response: {response.content}")


def example_quick_function():
    """Quick function example."""
    print("\n=== Quick Function ===")
    
    # One-liner
    response = complete("What is the speed of light?", provider="anthropic")
    print(f"Response: {response[:100]}...")


def main():
    """Run all examples."""
    print("=" * 60)
    print("SimpleChain Examples")
    print("=" * 60)
    
    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\nSet ANTHROPIC_API_KEY to run examples:")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        return
    
    try:
        example_simple_completion()
        example_with_messages()
        example_chain()
        example_router()
        example_agent()
        example_rag()
        example_conversation()
        example_parallel()
        asyncio.run(example_async())
        example_quick_function()
        
        print("\n" + "=" * 60)
        print("All examples completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
