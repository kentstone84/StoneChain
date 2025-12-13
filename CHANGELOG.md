# Changelog

All notable changes to StoneChain will be documented in this file.

## [1.0.0] - 2025-12-13

### Added
- Initial release 🪨

**Core (`stonechain.py`) - Zero Dependencies**
- LLM providers: Anthropic, OpenAI, Groq, Mistral, DeepSeek, Ollama
- Chain class for sequential prompts
- Agent class with tool calling
- RAG with built-in cosine similarity
- Conversation with memory
- Router for dynamic model selection
- Parallel execution support

**Vector Integrations (`stonechain_vectors.py`) - Optional**
- Pinecone integration
- Chroma integration
- Weaviate integration
- Qdrant integration
- Milvus integration
- PostgreSQL pgvector integration
- Embedding providers: OpenAI, Cohere, Voyage AI
- VectorRAG helper for production RAG

**MCP Client (`stonechain_mcp.py`) - Zero Dependencies**
- Stdio transport for local servers
- HTTP transport for remote servers
- Multi-server support
- Tool name prefixes
- Tool interceptors
- Resources and prompts support

**MCP Server (`stonechain_mcp_server.py`) - Zero Dependencies**
- Decorator-based tool registration
- Resource and prompt support
- Stdio and HTTP transports
- Auto-generated JSON schemas from type hints
- FastMCP replacement with zero dependencies

### Philosophy
- Core has zero external dependencies
- Extensions are optional and clearly separated
- Pure Python stdlib for core functionality
- Single file per concern
- Readable, maintainable code

**Built like a rock** 🪨
