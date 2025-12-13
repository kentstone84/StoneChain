#!/usr/bin/env python3
"""
Multi-Provider Example
======================

Shows how to use different providers and compare them.
"""

import os
import sys
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simplechain import (
    Anthropic, OpenAI, Groq, Mistral, DeepSeek, Ollama,
    Parallel, Message
)


def compare_providers():
    """Compare responses from different providers."""
    print("=" * 70)
    print("Multi-Provider Comparison")
    print("=" * 70)
    
    prompt = "Explain what makes a good API design in 2 sentences."
    
    providers = []
    
    # Add available providers
    if os.environ.get("ANTHROPIC_API_KEY"):
        providers.append(("Anthropic Claude", Anthropic()))
    
    if os.environ.get("OPENAI_API_KEY"):
        providers.append(("OpenAI GPT-4", OpenAI()))
    
    if os.environ.get("GROQ_API_KEY"):
        providers.append(("Groq Llama", Groq()))
    
    if os.environ.get("MISTRAL_API_KEY"):
        providers.append(("Mistral", Mistral()))
    
    if os.environ.get("DEEPSEEK_API_KEY"):
        providers.append(("DeepSeek", DeepSeek()))
    
    # Ollama always available if running
    try:
        ollama = Ollama()
        # Quick test
        providers.append(("Ollama Local", ollama))
    except:
        pass
    
    if not providers:
        print("\nNo providers available. Set API keys:")
        print("  export ANTHROPIC_API_KEY=...")
        print("  export OPENAI_API_KEY=...")
        print("  export GROQ_API_KEY=...")
        return
    
    print(f"\nPrompt: {prompt}")
    print(f"\nComparing {len(providers)} providers...\n")
    print("-" * 70)
    
    for name, llm in providers:
        try:
            response = llm.complete([Message.user(prompt)])
            print(f"\n{name}:")
            print(f"  Response: {response.content}")
            print(f"  Tokens: {response.total_tokens}")
            print(f"  Latency: {response.latency_ms:.0f}ms")
            print(f"  Speed: {response.tokens_per_second:.1f} tok/s")
        except Exception as e:
            print(f"\n{name}: Error - {e}")
    
    print("\n" + "-" * 70)


def parallel_comparison():
    """Run all providers in parallel."""
    print("\n" + "=" * 70)
    print("Parallel Execution")
    print("=" * 70)
    
    prompt = "What is 2+2? Answer with just the number."
    
    tasks = []
    names = []
    
    if os.environ.get("ANTHROPIC_API_KEY"):
        tasks.append((Anthropic(), prompt))
        names.append("Anthropic")
    
    if os.environ.get("OPENAI_API_KEY"):
        tasks.append((OpenAI(), prompt))
        names.append("OpenAI")
    
    if os.environ.get("GROQ_API_KEY"):
        tasks.append((Groq(), prompt))
        names.append("Groq")
    
    if len(tasks) < 2:
        print("\nNeed at least 2 providers for parallel comparison.")
        return
    
    print(f"\nRunning {len(tasks)} requests in parallel...")
    
    import time
    start = time.perf_counter()
    results = Parallel.run(tasks)
    total_time = (time.perf_counter() - start) * 1000
    
    print(f"\nTotal time: {total_time:.0f}ms (parallel)")
    print(f"Sum of individual: {sum(r.latency_ms for r in results):.0f}ms")
    print(f"Speedup: {sum(r.latency_ms for r in results) / total_time:.1f}x")
    
    for name, result in zip(names, results):
        print(f"\n{name}: {result.content} ({result.latency_ms:.0f}ms)")


def provider_fallback():
    """Demonstrate fallback between providers."""
    print("\n" + "=" * 70)
    print("Provider Fallback")
    print("=" * 70)
    
    providers = []
    
    # Order by preference
    if os.environ.get("ANTHROPIC_API_KEY"):
        providers.append(Anthropic())
    if os.environ.get("OPENAI_API_KEY"):
        providers.append(OpenAI())
    if os.environ.get("GROQ_API_KEY"):
        providers.append(Groq())
    
    if not providers:
        print("\nNo providers available.")
        return
    
    prompt = "Hello!"
    
    print(f"\nTrying {len(providers)} providers in order...")
    
    for i, llm in enumerate(providers):
        try:
            response = llm.complete([Message.user(prompt)])
            print(f"\n✓ Provider {i+1} ({llm.provider.value}) succeeded:")
            print(f"  {response.content}")
            break
        except Exception as e:
            print(f"\n✗ Provider {i+1} ({llm.provider.value}) failed: {e}")
    else:
        print("\nAll providers failed!")


def cost_estimation():
    """Estimate costs across providers."""
    print("\n" + "=" * 70)
    print("Cost Estimation")
    print("=" * 70)
    
    # Approximate costs per 1M tokens (as of late 2024)
    costs = {
        "anthropic": {"input": 3.00, "output": 15.00, "model": "claude-sonnet-4-20250514"},
        "openai": {"input": 5.00, "output": 15.00, "model": "gpt-4o"},
        "groq": {"input": 0.05, "output": 0.08, "model": "llama-3.3-70b"},
        "mistral": {"input": 2.00, "output": 6.00, "model": "mistral-large"},
        "deepseek": {"input": 0.14, "output": 0.28, "model": "deepseek-chat"},
    }
    
    # Simulate a workload
    input_tokens = 100000  # 100K input tokens
    output_tokens = 50000   # 50K output tokens
    
    print(f"\nEstimated costs for {input_tokens:,} input + {output_tokens:,} output tokens:\n")
    
    results = []
    for provider, info in costs.items():
        input_cost = (input_tokens / 1_000_000) * info["input"]
        output_cost = (output_tokens / 1_000_000) * info["output"]
        total = input_cost + output_cost
        results.append((total, provider, info["model"]))
    
    results.sort()
    
    for total, provider, model in results:
        print(f"  {provider:12} ({model:25}): ${total:.4f}")
    
    print(f"\nCheapest: {results[0][1]} (${results[0][0]:.4f})")
    print(f"Most expensive: {results[-1][1]} (${results[-1][0]:.4f})")
    print(f"Difference: {results[-1][0]/results[0][0]:.0f}x")


def main():
    compare_providers()
    parallel_comparison()
    provider_fallback()
    cost_estimation()
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
