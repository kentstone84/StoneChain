"""
StoneChain - Zero Dependency LLM Framework
Built like a rock.
"""

from .stonechain import (
    # Version
    __version__,
    __author__,
    __license__,
    
    # Core types
    Provider,
    Message,
    Response,
    Tool,
    Document,
    
    # Exceptions
    StoneChainError,
    APIError,
    ConfigError,
    
    # LLM adapters
    LLM,
    Anthropic,
    OpenAI,
    Ollama,
    Groq,
    Mistral,
    DeepSeek,
    
    # Orchestration
    Chain,
    Step,
    Router,
    Agent,
    RAG,
    
    # Memory
    Memory,
    Conversation,
    
    # Utilities
    Parallel,
    HTTP,
    http,
    
    # Convenience functions
    complete,
    acomplete,
)

__all__ = [
    # Version
    "__version__",
    "__author__",
    "__license__",
    
    # Core types
    "Provider",
    "Message",
    "Response",
    "Tool",
    "Document",
    
    # Exceptions
    "StoneChainError",
    "APIError",
    "ConfigError",
    
    # LLM adapters
    "LLM",
    "Anthropic",
    "OpenAI",
    "Ollama",
    "Groq",
    "Mistral",
    "DeepSeek",
    
    # Orchestration
    "Chain",
    "Step",
    "Router",
    "Agent",
    "RAG",
    
    # Memory
    "Memory",
    "Conversation",
    
    # Utilities
    "Parallel",
    "HTTP",
    "http",
    
    # Convenience
    "complete",
    "acomplete",
]
