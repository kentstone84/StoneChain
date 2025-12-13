# Changelog

All notable changes to SimpleChain will be documented here.

## [1.0.0] - 2025-12-13

### Added
- Initial release
- Zero-dependency LLM framework
- Provider adapters:
  - Anthropic Claude
  - OpenAI GPT
  - Groq
  - Mistral
  - DeepSeek
  - Ollama (local)
- Core features:
  - Chain: Sequential LLM calls with variable passing
  - Router: Route inputs to different LLMs/chains
  - Agent: Tool-using agent with ReAct reasoning
  - RAG: Simple document Q&A
  - Memory: Conversation history management
  - Parallel: Concurrent LLM calls
- Convenience functions: `complete()`, `acomplete()`
- Full async support
- Comprehensive test suite
- MIT License

### Philosophy
- Pure Python stdlib only
- No external dependencies
- Single file distribution possible
- Readable in 30 minutes
