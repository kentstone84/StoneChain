# StoneChain

**The zero-dependency LLM framework. LangChain in 800 lines. Built like a rock.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Core Dependencies: None](https://img.shields.io/badge/core%20deps-none-green.svg)](https://github.com/KentStone/stonechain)

## Why?

LangChain is bloated. 200+ dependencies. 100,000+ lines. Abstraction hell.

StoneChain does the same thing in **one file** with **zero dependencies**.

| | LangChain | StoneChain |
|--|-----------|------------|
| Core dependencies | 200+ | **0** |
| Install size | 50MB+ | **36KB** |
| Lines of code | 100,000+ | **~800** |
| Time to understand | Days | **Minutes** |

## Install

```bash
# Option 1: Copy the file (recommended)
curl -O https://raw.githubusercontent.com/KentStone/stonechain/main/stonechain.py

# Option 2: pip
pip install stonechain
```

## Quick Start

```python
from stonechain import Anthropic

# That's it. No config. No setup.
llm = Anthropic()  # Uses ANTHROPIC_API_KEY env var
print(llm("What is 2+2?"))  # "4"
```

## Providers

```python
from stonechain import Anthropic, OpenAI, Groq, Mistral, DeepSeek, Ollama

# Cloud providers (need API keys)
llm = Anthropic()                    # claude-sonnet-4-20250514
llm = OpenAI()                       # gpt-4o
llm = Groq()                         # llama-3.3-70b-versatile
llm = Mistral()                      # mistral-large-latest
llm = DeepSeek()                     # deepseek-chat

# Local (no API key needed)
llm = Ollama(model="llama3.2")       # Any Ollama model
```

## Core Features

### Chain (Sequential Calls)

```python
from stonechain import Anthropic, Chain

chain = Chain(Anthropic())
chain.add("analyze", "Analyze: {input}", "analysis")
chain.add("critique", "Critique: {analysis}", "critique")

result = chain.run(input="AI safety")
```

### Agent (Tool Use)

```python
from stonechain import Anthropic, Agent, Tool

def calculator(expression: str) -> str:
    return str(eval(expression))

agent = Agent(Anthropic(), [
    Tool("calculator", "Do math", {"expression": {"type": "string"}}, calculator)
])
result = agent.run("What is 15 * 23?")
```

### RAG (Document Q&A)

```python
from stonechain import Anthropic, RAG, Document

rag = RAG(Anthropic())
rag.add([Document("StoneChain was created by Kent Stone.")])
answer = rag.query("Who created StoneChain?")
```

### Conversation (Memory)

```python
from stonechain import Anthropic, Conversation

conv = Conversation(Anthropic(), system="You are a pirate.")
print(conv.chat("Hello!"))  # "Ahoy, matey!"
```

## Vector Database Integrations

For production RAG, use `stonechain_vectors.py` with your preferred vector DB:

```python
from stonechain import Anthropic
from stonechain_vectors import VectorRAG, ChromaStore, OpenAIEmbeddings

# Production RAG with Chroma + OpenAI embeddings
rag = VectorRAG(
    llm=Anthropic(),
    store=ChromaStore(persist_directory="./my_db"),
    embeddings=OpenAIEmbeddings()
)

rag.add(["Document 1", "Document 2", "Document 3"])
answer = rag.query("What's in document 1?")
```

### Supported Vector Databases

| Database | Install | Usage |
|----------|---------|-------|
| **Pinecone** | `pip install pinecone-client` | `PineconeStore(api_key="...", index_name="...")` |
| **Chroma** | `pip install chromadb` | `ChromaStore(persist_directory="./db")` |
| **Weaviate** | `pip install weaviate-client` | `WeaviateStore(url="http://localhost:8080")` |
| **Qdrant** | `pip install qdrant-client` | `QdrantStore(url="http://localhost:6333")` |
| **Milvus** | `pip install pymilvus` | `MilvusStore(host="localhost")` |
| **PostgreSQL** | `pip install psycopg2-binary pgvector` | `PgVectorStore(connection_string="...")` |

### Supported Embedding Providers

| Provider | Env Var | Usage |
|----------|---------|-------|
| **OpenAI** | `OPENAI_API_KEY` | `OpenAIEmbeddings(model="text-embedding-3-small")` |
| **Cohere** | `COHERE_API_KEY` | `CohereEmbeddings(model="embed-english-v3.0")` |
| **Voyage AI** | `VOYAGE_API_KEY` | `VoyageEmbeddings(model="voyage-2")` |

## MCP Support (Model Context Protocol)

StoneChain includes **zero-dependency** MCP support:

### MCP Client

```python
from stonechain import Anthropic, Agent
from stonechain_mcp import MCPClient, StdioTransport, HTTPTransport

client = MCPClient({
    "math": StdioTransport("python", ["math_server.py"]),
    "weather": HTTPTransport("http://localhost:8000/mcp"),
})

tools = await client.get_tools()
agent = Agent(Anthropic(), tools)
```

### MCP Server (FastMCP Alternative)

```python
from stonechain_mcp_server import MCPServer

server = MCPServer("Math")

@server.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

server.run()
```

## File Structure

```
stonechain/
├── stonechain.py          # Core (zero deps) - LLM, Chain, Agent, RAG
├── stonechain_vectors.py  # Vector DBs (optional deps) - Pinecone, Chroma, etc.
├── stonechain_mcp.py      # MCP client (zero deps)
└── stonechain_mcp_server.py # MCP server (zero deps)
```

**Philosophy**: Core has zero dependencies. Extensions are optional.

## Environment Variables

```bash
# LLM Providers
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
MISTRAL_API_KEY=...
DEEPSEEK_API_KEY=...

# Embedding Providers (for stonechain_vectors)
COHERE_API_KEY=...
VOYAGE_API_KEY=...
```

## Documentation

- [Migration Guide](docs/MIGRATION.md) - Moving from LangChain
- [Advanced Usage](docs/ADVANCED.md) - Custom providers, agents, RAG

## Why "StoneChain"?

1. **Stone** - Built solid. No flaky dependencies.
2. **Stone** - The author's name (Kent Stone)
3. **Chain** - LLM orchestration

## License

MIT License - do whatever you want.

## Author

Kent Stone ([@KentStone](https://github.com/KentStone))

Creator of [JARVIS Cognitive AI](https://github.com/KentStone/jarvis-pro) and the Stone Retrieval Function (SRF).

---

**Built like a rock. Zero dependencies. Zero excuses.** 🪨
