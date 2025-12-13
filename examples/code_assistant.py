#!/usr/bin/env python3
"""
Build a Code Assistant
======================

Example of building a coding assistant with StoneChain.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stonechain import Anthropic, Chain, Agent, Tool, Conversation


def code_reviewer():
    """Review code for issues."""
    print("=" * 60)
    print("Code Reviewer")
    print("=" * 60)
    
    llm = Anthropic()
    
    chain = Chain(llm, system="""You are an expert code reviewer.
Focus on: bugs, security issues, performance, readability.
Be constructive and specific.""")
    
    chain.add("review", """Review this code:

```
{code}
```

Provide:
1. Summary (1 sentence)
2. Issues found (if any)
3. Suggestions for improvement
""", "review")
    
    code = '''
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    result = db.execute(query)
    return result[0]
'''
    
    print(f"\nCode to review:\n{code}")
    print("\nReview:")
    print("-" * 40)
    
    result = chain.run(code=code)
    print(result["outputs"]["review"])


def code_generator():
    """Generate code from descriptions."""
    print("\n" + "=" * 60)
    print("Code Generator")
    print("=" * 60)
    
    llm = Anthropic()
    
    chain = Chain(llm, system="""You are an expert programmer.
Write clean, well-documented, production-ready code.
Include error handling and type hints where appropriate.""")
    
    chain.add("code", """Write Python code for: {description}

Requirements:
- Clean and readable
- Include docstrings
- Handle edge cases
- Use type hints

Output only the code, no explanations.""", "code")
    
    chain.add("test", """Write pytest tests for this code:

{code}

Include:
- Happy path tests
- Edge case tests
- Error handling tests""", "tests")
    
    description = "a function that validates email addresses"
    
    print(f"\nDescription: {description}")
    print("\nGenerating code and tests...")
    print("-" * 40)
    
    result = chain.run(description=description)
    
    print("\nGenerated Code:")
    print(result["outputs"]["code"])
    print("\nGenerated Tests:")
    print(result["outputs"]["tests"])


def code_explainer():
    """Explain code in simple terms."""
    print("\n" + "=" * 60)
    print("Code Explainer")
    print("=" * 60)
    
    llm = Anthropic()
    conv = Conversation(llm, system="""You are a patient programming teacher.
Explain code clearly and simply. Use analogies when helpful.
Ask if the user wants more detail on any part.""")
    
    code = '''
@lru_cache(maxsize=128)
def fibonacci(n: int) -> int:
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
'''
    
    print(f"\nCode to explain:\n{code}")
    print("\nExplanation:")
    print("-" * 40)
    
    response = conv.chat(f"Explain this code:\n{code}")
    print(response)
    
    print("\n" + "-" * 40)
    response = conv.chat("What does lru_cache do?")
    print(f"\nFollow-up about lru_cache:\n{response}")


def code_debugger():
    """Debug code with tools."""
    print("\n" + "=" * 60)
    print("Code Debugger")
    print("=" * 60)
    
    def run_python(code: str) -> str:
        """Safely execute Python code."""
        try:
            allowed_builtins = {
                'print': print,
                'len': len,
                'range': range,
                'str': str,
                'int': int,
                'float': float,
                'list': list,
                'dict': dict,
                'sum': sum,
                'min': min,
                'max': max,
                'abs': abs,
                'round': round,
            }
            
            local_vars = {}
            exec(code, {"__builtins__": allowed_builtins}, local_vars)
            return f"Executed successfully. Variables: {local_vars}"
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"
    
    def analyze_error(error: str) -> str:
        """Analyze Python error messages."""
        if "SyntaxError" in error:
            return "Syntax error - check for missing colons, parentheses, or indentation"
        elif "NameError" in error:
            return "Variable or function not defined - check spelling and scope"
        elif "TypeError" in error:
            return "Type mismatch - check argument types and operations"
        elif "IndexError" in error:
            return "List index out of range - check array bounds"
        elif "KeyError" in error:
            return "Dictionary key not found - check key existence"
        else:
            return "Unknown error type"
    
    tools = [
        Tool(
            "run_code",
            "Execute Python code and return the result",
            {"code": {"type": "string", "description": "Python code to execute"}},
            run_python
        ),
        Tool(
            "analyze_error",
            "Analyze a Python error message",
            {"error": {"type": "string", "description": "Error message"}},
            analyze_error
        ),
    ]
    
    llm = Anthropic()
    agent = Agent(llm, tools, max_iterations=10, system="""You are a debugging expert.
When debugging:
1. First try to run the code to see the error
2. Analyze the error
3. Suggest a fix
4. Verify the fix works

Be methodical and explain your reasoning.""")
    
    buggy_code = '''
def factorial(n):
    if n = 0:
        return 1
    return n * factorial(n - 1)

print(factorial(5))
'''
    
    print(f"\nBuggy code:\n{buggy_code}")
    print("\nDebugging...")
    print("-" * 40)
    
    result = agent.run(f"Debug this code and fix any issues:\n{buggy_code}")
    print(f"\nResult: {result.get('answer', 'No answer')}")


def code_translator():
    """Translate code between languages."""
    print("\n" + "=" * 60)
    print("Code Translator")
    print("=" * 60)
    
    llm = Anthropic()
    
    chain = Chain(llm, system="""You are a polyglot programmer expert in all languages.
Translate code while maintaining:
- Logic and structure
- Idiomatic patterns for the target language
- Comments explaining language-specific differences""")
    
    chain.add("translate", """Translate this {source_lang} code to {target_lang}:

```{source_lang}
{code}
```

Output only the translated code with comments.""", "translated")
    
    python_code = '''
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
'''
    
    print(f"\nPython code:\n{python_code}")
    print("\nTranslating to JavaScript...")
    print("-" * 40)
    
    result = chain.run(
        code=python_code,
        source_lang="Python",
        target_lang="JavaScript"
    )
    
    print(result["outputs"]["translated"])


def main():
    """Run code assistant examples."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY to run examples")
        return
    
    code_reviewer()
    code_generator()
    code_explainer()
    code_debugger()
    code_translator()
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
