# SimpleChain

**The zero-dependency LLM framework. LangChain in 800 lines.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Dependencies: None](https://img.shields.io/badge/dependencies-none-green.svg)](https://github.com/KentStone/simplechain)

## Why?

LangChain is bloated. 200+ dependencies. 100,000+ lines. Abstraction hell.

SimpleChain does the same thing in **one file** with **zero dependencies**.

| | LangChain | SimpleChain |
|--|-----------|-------------|
| Dependencies | 200+ | **0** |
| Install size | 50MB+ | **36KB** |
| Lines of code | 100,000+ | **~800** |
| Time to understand | Days | **Minutes** |

## Install

```bash
# Option 1: Copy the file (recommended)
curl -O https://raw.githubusercontent.com/KentStone/simplechain/main/simplechain.py

# Option 2: pip (coming soon)
pip install simplechain
```

## Quick Start

```python
from simplechain import Anthropic

# That's it. No config. No setup.
llm = Anthropic()  # Uses ANTHROPIC_API_KEY env var
print(llm("What is 2+2?"))  # "4"
```

## Providers

```python
from simplechain import Anthropic, OpenAI, Groq, Mistral, DeepSeek, Ollama

# Cloud providers (need API keys)
llm = Anthropic()                    # claude-sonnet-4-20250514
llm = OpenAI()                       # gpt-4o
llm = Groq()                         # llama-3.3-70b-versatile
llm = Mistral()                      # mistral-large-latest
llm = DeepSeek()                     # deepseek-chat

# Local (no API key needed)
llm = Ollama(model="llama3.2")       # Any Ollama model
```

## Features

### Simple Completion

```python
from simplechain import Anthropic, Message

llm = Anthropic()

# Shorthand
response = llm("Explain quantum computing")

# Full control
response = llm.complete(
    messages=[
        Message.system("You are a physicist."),
        Message.user("Explain quantum computing")
    ],
    temperature=0.3,
    max_tokens=500
)
print(response.content)
print(f"Tokens: {response.total_tokens}")
print(f"Latency: {response.latency_ms}ms")
```

### Chain (Sequential Calls)

```python
from simplechain import Anthropic, Chain

llm = Anthropic()

chain = Chain(llm, system="You are a helpful assistant.")
chain.add("analyze", "Analyze this topic: {input}", "analysis")
chain.add("critique", "Critique this analysis: {analysis}", "critique")
chain.add("synthesize", "Synthesize:\n{analysis}\n{critique}", "final")

result = chain.run(input="Artificial Intelligence")
print(result["outputs"]["final"])
```

### Router

```python
from simplechain import Anthropic, Groq, Router

claude = Anthropic()
groq = Groq()  # Fast but less capable

router = Router(default=claude)
router.add("simple", lambda x: len(x) < 50, groq)  # Short queries -> Groq
router.add("code", lambda x: "code" in x.lower(), claude)  # Code -> Claude

result = router.route("Hi")  # -> Groq (fast)
result = router.route("Write a complex algorithm")  # -> Claude (smart)
```

### Agent (Tool Use)

```python
from simplechain import Anthropic, Agent, Tool

def calculator(expression: str) -> str:
    return str(eval(expression))

def search(query: str) -> str:
    return f"Results for: {query}"

tools = [
    Tool("calculator", "Do math", {"expression": {"type": "string"}}, calculator),
    Tool("search", "Search web", {"query": {"type": "string"}}, search),
]

agent = Agent(Anthropic(), tools)
result = agent.run("What is 15 * 23 + 7?")
print(result["answer"])  # "352"
```

### RAG (Document Q&A)

```python
from simplechain import Anthropic, RAG, Document

rag = RAG(Anthropic())

rag.add([
    Document("SimpleChain was created by Kent Stone in 2025."),
    Document("It has zero dependencies and replaces LangChain."),
    Document("The codebase is only 800 lines of Python."),
])

result = rag.query("Who created SimpleChain?")
print(result["answer"])  # "Kent Stone created SimpleChain in 2025."
```

### Conversation (Memory)

```python
from simplechain import Anthropic, Conversation

conv = Conversation(Anthropic(), system="You are a pirate.")

print(conv.chat("Hello!"))           # "Ahoy, matey!"
print(conv.chat("What's your name?")) # "They call me Captain Claude!"
print(conv.chat("What did I just ask?"))  # Remembers context
```

### Parallel Execution

```python
from simplechain import Anthropic, OpenAI, Parallel

claude = Anthropic()
gpt = OpenAI()

# Run in parallel
results = Parallel.run([
    (claude, "Explain AI"),
    (gpt, "Explain AI"),
])

for r in results:
    print(f"{r.provider}: {r.content[:100]}...")
```

### Quick Functions

```python
from simplechain import complete, acomplete

# Sync
response = complete("Hello!", provider="anthropic")

# Async
response = await acomplete("Hello!", provider="openai")
```

## API Reference

### LLM Classes

All LLM classes share the same interface:

```python
class LLM:
    def complete(messages, model=None, temperature=0.7, max_tokens=4096) -> Response
    async def acomplete(...) -> Response
    def __call__(prompt) -> str  # Shorthand
```

### Response Object

```python
@dataclass
class Response:
    content: str           # The response text
    model: str            # Model used
    provider: Provider    # Provider enum
    input_tokens: int     # Tokens in prompt
    output_tokens: int    # Tokens in response
    latency_ms: float     # Time taken
    finish_reason: str    # Why it stopped
    raw: dict            # Raw API response
    
    @property
    def total_tokens(self) -> int
    
    @property
    def tokens_per_second(self) -> float
```

### Message Helpers

```python
Message.system("You are helpful.")
Message.user("Hello!")
Message.assistant("Hi there!")
```

### Tool Definition

```python
Tool(
    name="calculator",
    description="Do math calculations",
    parameters={"expression": {"type": "string", "description": "Math expression"}},
    function=lambda expression: eval(expression)
)
```

## Environment Variables

```bash
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
MISTRAL_API_KEY=...
DEEPSEEK_API_KEY=...
```

## How It Works

Pure Python stdlib:
- `urllib.request` - HTTP calls
- `json` - Parsing
- `dataclasses` - Types
- `asyncio` - Async support
- `ssl` - HTTPS
- `concurrent.futures` - Parallelism

**No requests. No httpx. No aiohttp. Nothing external.**

## Contributing

1. Fork the repo
2. Make changes to `simplechain.py`
3. Run tests: `python -m pytest tests/`
4. Submit PR

Keep it simple. No new dependencies. Ever.

## License

MIT License - do whatever you want.

## Author

Kent Stone ([@KentStone](https://github.com/KentStone))

---

**Zero dependencies. Zero excuses. Just works.**
