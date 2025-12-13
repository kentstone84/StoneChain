#!/usr/bin/env python3
"""
SimpleChain - The Zero-Dependency LLM Framework
================================================

LangChain replacement in 600 lines. Pure Python stdlib.
No pip install hell. No abstraction madness. Just works.

GitHub: https://github.com/KentStone/simplechain
License: MIT

Author: Kent Stone
"""

__version__ = "1.0.0"
__author__ = "Kent Stone"
__license__ = "MIT"

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union, TypeVar, Generic
from enum import Enum
from abc import ABC, abstractmethod
import urllib.request
import urllib.error
import json
import time
import ssl
import asyncio
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache


# =============================================================================
# CORE TYPES
# =============================================================================

class Provider(Enum):
    """Supported LLM providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"
    GROQ = "groq"
    MISTRAL = "mistral"
    COHERE = "cohere"
    TOGETHER = "together"
    ANYSCALE = "anyscale"
    FIREWORKS = "fireworks"
    DEEPSEEK = "deepseek"
    CUSTOM = "custom"


@dataclass
class Message:
    """A chat message."""
    role: str  # "system", "user", "assistant", "tool"
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        d = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d
    
    @classmethod
    def system(cls, content: str) -> "Message":
        return cls("system", content)
    
    @classmethod
    def user(cls, content: str) -> "Message":
        return cls("user", content)
    
    @classmethod
    def assistant(cls, content: str) -> "Message":
        return cls("assistant", content)


@dataclass
class Response:
    """LLM response with metadata."""
    content: str
    model: str
    provider: Provider
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0
    finish_reason: Optional[str] = None
    raw: Optional[Dict] = None
    
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
    
    @property
    def tokens_per_second(self) -> float:
        if self.latency_ms > 0:
            return (self.output_tokens / self.latency_ms) * 1000
        return 0


@dataclass
class Tool:
    """A callable tool for agents."""
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable[..., Any]
    
    def to_openai_schema(self) -> Dict[str, Any]:
        """OpenAI-style function schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": list(self.parameters.keys())
                }
            }
        }
    
    def to_anthropic_schema(self) -> Dict[str, Any]:
        """Anthropic-style tool schema."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.parameters,
                "required": list(self.parameters.keys())
            }
        }
    
    def __call__(self, **kwargs) -> Any:
        return self.function(**kwargs)


@dataclass
class Document:
    """A document for RAG."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = hashlib.md5(self.content.encode()).hexdigest()[:12]


# =============================================================================
# HTTP CLIENT - Pure stdlib, zero dependencies
# =============================================================================

class HTTP:
    """
    Pure stdlib HTTP client.
    No requests. No httpx. No aiohttp. Just urllib.
    """
    
    _instance = None
    _ssl_context = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._ssl_context = ssl.create_default_context()
        return cls._instance
    
    def post(
        self,
        url: str,
        headers: Dict[str, str],
        data: Dict[str, Any],
        timeout: float = 120.0
    ) -> Dict[str, Any]:
        """Synchronous POST."""
        json_data = json.dumps(data).encode('utf-8')
        
        req = urllib.request.Request(
            url,
            data=json_data,
            headers=headers,
            method='POST'
        )
        
        try:
            with urllib.request.urlopen(
                req,
                timeout=timeout,
                context=self._ssl_context
            ) as resp:
                return json.loads(resp.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            body = e.read().decode('utf-8')
            raise APIError(f"HTTP {e.code}: {body}", e.code)
        except urllib.error.URLError as e:
            raise ConnectionError(f"Connection failed: {e.reason}")
    
    async def post_async(
        self,
        url: str,
        headers: Dict[str, str],
        data: Dict[str, Any],
        timeout: float = 120.0
    ) -> Dict[str, Any]:
        """Async POST via thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.post(url, headers, data, timeout)
        )


# Global HTTP instance
http = HTTP()


# =============================================================================
# EXCEPTIONS
# =============================================================================

class SimpleChainError(Exception):
    """Base exception."""
    pass


class APIError(SimpleChainError):
    """API call failed."""
    def __init__(self, message: str, status_code: int = None):
        super().__init__(message)
        self.status_code = status_code


class ConfigError(SimpleChainError):
    """Configuration error."""
    pass


# =============================================================================
# LLM ADAPTERS
# =============================================================================

class LLM(ABC):
    """Base LLM adapter."""
    
    provider: Provider
    
    @abstractmethod
    def complete(
        self,
        messages: List[Message],
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stop: List[str] = None,
        **kwargs
    ) -> Response:
        """Generate completion synchronously."""
        pass
    
    async def acomplete(
        self,
        messages: List[Message],
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stop: List[str] = None,
        **kwargs
    ) -> Response:
        """Generate completion asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.complete(
                messages, model, temperature, max_tokens, stop, **kwargs
            )
        )
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """Shorthand for simple completion."""
        response = self.complete([Message.user(prompt)], **kwargs)
        return response.content


class Anthropic(LLM):
    """
    Anthropic Claude adapter.
    
    Usage:
        llm = Anthropic(api_key)
        response = llm.complete([Message.user("Hello")])
    """
    
    provider = Provider.ANTHROPIC
    
    def __init__(
        self,
        api_key: str = None,
        model: str = "claude-sonnet-4-20250514",
        base_url: str = "https://api.anthropic.com/v1"
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ConfigError("ANTHROPIC_API_KEY required")
        self.default_model = model
        self.base_url = base_url
    
    def complete(
        self,
        messages: List[Message],
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stop: List[str] = None,
        **kwargs
    ) -> Response:
        start = time.perf_counter()
        
        # Separate system message (Anthropic style)
        system = ""
        chat_messages = []
        for msg in messages:
            if msg.role == "system":
                system = msg.content
            else:
                chat_messages.append(msg.to_dict())
        
        payload = {
            "model": model or self.default_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": chat_messages
        }
        
        if system:
            payload["system"] = system
        if stop:
            payload["stop_sequences"] = stop
        
        data = http.post(
            f"{self.base_url}/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            data=payload
        )
        
        latency = (time.perf_counter() - start) * 1000
        
        content = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                content += block.get("text", "")
        
        return Response(
            content=content,
            model=model or self.default_model,
            provider=self.provider,
            input_tokens=data.get("usage", {}).get("input_tokens", 0),
            output_tokens=data.get("usage", {}).get("output_tokens", 0),
            latency_ms=latency,
            finish_reason=data.get("stop_reason"),
            raw=data
        )


class OpenAI(LLM):
    """
    OpenAI GPT adapter. Also works with OpenAI-compatible APIs.
    
    Usage:
        llm = OpenAI(api_key)
        response = llm.complete([Message.user("Hello")])
    """
    
    provider = Provider.OPENAI
    
    def __init__(
        self,
        api_key: str = None,
        model: str = "gpt-4o",
        base_url: str = "https://api.openai.com/v1"
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ConfigError("OPENAI_API_KEY required")
        self.default_model = model
        self.base_url = base_url
    
    def complete(
        self,
        messages: List[Message],
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stop: List[str] = None,
        **kwargs
    ) -> Response:
        start = time.perf_counter()
        
        payload = {
            "model": model or self.default_model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if stop:
            payload["stop"] = stop
        
        data = http.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            data=payload
        )
        
        latency = (time.perf_counter() - start) * 1000
        choice = data.get("choices", [{}])[0]
        
        return Response(
            content=choice.get("message", {}).get("content", ""),
            model=model or self.default_model,
            provider=self.provider,
            input_tokens=data.get("usage", {}).get("prompt_tokens", 0),
            output_tokens=data.get("usage", {}).get("completion_tokens", 0),
            latency_ms=latency,
            finish_reason=choice.get("finish_reason"),
            raw=data
        )


class Ollama(LLM):
    """
    Local Ollama adapter.
    
    Usage:
        llm = Ollama(model="llama3.2")
        response = llm.complete([Message.user("Hello")])
    """
    
    provider = Provider.OLLAMA
    
    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434"
    ):
        self.default_model = model
        self.base_url = base_url
    
    def complete(
        self,
        messages: List[Message],
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stop: List[str] = None,
        **kwargs
    ) -> Response:
        start = time.perf_counter()
        
        payload = {
            "model": model or self.default_model,
            "messages": [m.to_dict() for m in messages],
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            },
            "stream": False
        }
        
        if stop:
            payload["options"]["stop"] = stop
        
        data = http.post(
            f"{self.base_url}/api/chat",
            headers={"Content-Type": "application/json"},
            data=payload,
            timeout=300.0  # Local models can be slow
        )
        
        latency = (time.perf_counter() - start) * 1000
        
        return Response(
            content=data.get("message", {}).get("content", ""),
            model=model or self.default_model,
            provider=self.provider,
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            latency_ms=latency,
            raw=data
        )


class Groq(LLM):
    """
    Groq adapter (fast inference).
    
    Usage:
        llm = Groq(api_key)
        response = llm.complete([Message.user("Hello")])
    """
    
    provider = Provider.GROQ
    
    def __init__(
        self,
        api_key: str = None,
        model: str = "llama-3.3-70b-versatile"
    ):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ConfigError("GROQ_API_KEY required")
        self.default_model = model
        self.base_url = "https://api.groq.com/openai/v1"
    
    def complete(
        self,
        messages: List[Message],
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stop: List[str] = None,
        **kwargs
    ) -> Response:
        start = time.perf_counter()
        
        payload = {
            "model": model or self.default_model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if stop:
            payload["stop"] = stop
        
        data = http.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            data=payload
        )
        
        latency = (time.perf_counter() - start) * 1000
        choice = data.get("choices", [{}])[0]
        
        return Response(
            content=choice.get("message", {}).get("content", ""),
            model=model or self.default_model,
            provider=self.provider,
            input_tokens=data.get("usage", {}).get("prompt_tokens", 0),
            output_tokens=data.get("usage", {}).get("completion_tokens", 0),
            latency_ms=latency,
            finish_reason=choice.get("finish_reason"),
            raw=data
        )


class Mistral(LLM):
    """Mistral AI adapter."""
    
    provider = Provider.MISTRAL
    
    def __init__(
        self,
        api_key: str = None,
        model: str = "mistral-large-latest"
    ):
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not self.api_key:
            raise ConfigError("MISTRAL_API_KEY required")
        self.default_model = model
        self.base_url = "https://api.mistral.ai/v1"
    
    def complete(
        self,
        messages: List[Message],
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stop: List[str] = None,
        **kwargs
    ) -> Response:
        start = time.perf_counter()
        
        payload = {
            "model": model or self.default_model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if stop:
            payload["stop"] = stop
        
        data = http.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            data=payload
        )
        
        latency = (time.perf_counter() - start) * 1000
        choice = data.get("choices", [{}])[0]
        
        return Response(
            content=choice.get("message", {}).get("content", ""),
            model=model or self.default_model,
            provider=self.provider,
            input_tokens=data.get("usage", {}).get("prompt_tokens", 0),
            output_tokens=data.get("usage", {}).get("completion_tokens", 0),
            latency_ms=latency,
            finish_reason=choice.get("finish_reason"),
            raw=data
        )


class DeepSeek(LLM):
    """DeepSeek adapter."""
    
    provider = Provider.DEEPSEEK
    
    def __init__(
        self,
        api_key: str = None,
        model: str = "deepseek-chat"
    ):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ConfigError("DEEPSEEK_API_KEY required")
        self.default_model = model
        self.base_url = "https://api.deepseek.com/v1"
    
    def complete(
        self,
        messages: List[Message],
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stop: List[str] = None,
        **kwargs
    ) -> Response:
        start = time.perf_counter()
        
        payload = {
            "model": model or self.default_model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if stop:
            payload["stop"] = stop
        
        data = http.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            data=payload
        )
        
        latency = (time.perf_counter() - start) * 1000
        choice = data.get("choices", [{}])[0]
        
        return Response(
            content=choice.get("message", {}).get("content", ""),
            model=model or self.default_model,
            provider=self.provider,
            input_tokens=data.get("usage", {}).get("prompt_tokens", 0),
            output_tokens=data.get("usage", {}).get("completion_tokens", 0),
            latency_ms=latency,
            finish_reason=choice.get("finish_reason"),
            raw=data
        )


# =============================================================================
# CHAIN - Sequential LLM calls
# =============================================================================

@dataclass
class Step:
    """A chain step."""
    name: str
    template: str
    output_key: str
    llm: Optional[LLM] = None
    temperature: Optional[float] = None
    
    def format(self, context: Dict[str, Any]) -> str:
        """Format template with context variables."""
        result = self.template
        for key, value in context.items():
            result = result.replace(f"{{{key}}}", str(value))
        return result


class Chain:
    """
    Sequential chain of LLM calls.
    
    Usage:
        chain = Chain(llm)
        chain.add("analyze", "Analyze: {input}", "analysis")
        chain.add("summarize", "Summarize: {analysis}", "summary")
        result = chain.run(input="Hello world")
    """
    
    def __init__(self, llm: LLM, system: str = None):
        self.llm = llm
        self.system = system
        self.steps: List[Step] = []
    
    def add(
        self,
        name: str,
        template: str,
        output_key: str,
        llm: LLM = None,
        temperature: float = None
    ) -> "Chain":
        """Add a step."""
        self.steps.append(Step(name, template, output_key, llm, temperature))
        return self
    
    def run(self, **inputs) -> Dict[str, Any]:
        """Run synchronously."""
        context = dict(inputs)
        results = {
            "steps": [],
            "total_tokens": 0,
            "total_latency_ms": 0.0
        }
        
        for step in self.steps:
            messages = []
            if self.system:
                messages.append(Message.system(self.system))
            messages.append(Message.user(step.format(context)))
            
            llm = step.llm or self.llm
            response = llm.complete(
                messages,
                temperature=step.temperature or 0.7
            )
            
            context[step.output_key] = response.content
            results["steps"].append({
                "name": step.name,
                "output": response.content,
                "tokens": response.total_tokens,
                "latency_ms": response.latency_ms
            })
            results["total_tokens"] += response.total_tokens
            results["total_latency_ms"] += response.latency_ms
        
        results["outputs"] = {s.output_key: context[s.output_key] for s in self.steps}
        return results
    
    async def arun(self, **inputs) -> Dict[str, Any]:
        """Run asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.run(**inputs))


# =============================================================================
# ROUTER - Route to different LLMs/chains
# =============================================================================

class Router:
    """
    Route inputs to different handlers.
    
    Usage:
        router = Router(default_llm)
        router.add("code", lambda x: "code" in x, code_llm)
        response = router.route("Write python code")
    """
    
    def __init__(self, default: Union[LLM, Chain] = None):
        self.default = default
        self.routes: List[tuple] = []  # (name, condition, handler, priority)
    
    def add(
        self,
        name: str,
        condition: Callable[[str], bool],
        handler: Union[LLM, Chain],
        priority: int = 0
    ) -> "Router":
        """Add a route."""
        self.routes.append((name, condition, handler, priority))
        self.routes.sort(key=lambda r: -r[3])  # Sort by priority desc
        return self
    
    def route(self, input_text: str, **kwargs) -> Dict[str, Any]:
        """Route input to appropriate handler."""
        for name, condition, handler, _ in self.routes:
            if condition(input_text):
                if isinstance(handler, Chain):
                    result = handler.run(input=input_text, **kwargs)
                else:
                    response = handler.complete([Message.user(input_text)])
                    result = {"response": response.content}
                result["routed_to"] = name
                return result
        
        if self.default:
            if isinstance(self.default, Chain):
                result = self.default.run(input=input_text, **kwargs)
            else:
                response = self.default.complete([Message.user(input_text)])
                result = {"response": response.content}
            result["routed_to"] = "default"
            return result
        
        raise ValueError("No matching route")


# =============================================================================
# AGENT - Tool-using agent
# =============================================================================

class Agent:
    """
    Tool-using agent with ReAct-style reasoning.
    
    Usage:
        agent = Agent(llm, tools=[calculator, search])
        result = agent.run("What is 15 * 23?")
    """
    
    def __init__(
        self,
        llm: LLM,
        tools: List[Tool],
        max_iterations: int = 10,
        system: str = None
    ):
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.max_iterations = max_iterations
        self.system = system or self._default_system()
    
    def _default_system(self) -> str:
        tool_desc = "\n".join(f"- {t.name}: {t.description}" for t in self.tools.values())
        return f"""You are a helpful assistant with tools.

Available tools:
{tool_desc}

To use a tool:
TOOL: <name>
INPUT: <json>

When done:
FINAL: <answer>
"""
    
    def run(self, query: str) -> Dict[str, Any]:
        """Run synchronously."""
        messages = [Message.system(self.system), Message.user(query)]
        iterations = []
        
        for i in range(self.max_iterations):
            response = self.llm.complete(messages)
            content = response.content
            iteration = {"step": i + 1, "response": content}
            
            if "FINAL:" in content:
                answer = content.split("FINAL:")[-1].strip()
                iteration["type"] = "final"
                iterations.append(iteration)
                return {"answer": answer, "iterations": iterations}
            
            if "TOOL:" in content and "INPUT:" in content:
                tool_name = content.split("TOOL:")[-1].split("\n")[0].strip()
                input_str = content.split("INPUT:")[-1].strip()
                
                try:
                    tool_input = json.loads(input_str)
                except:
                    tool_input = {"raw": input_str}
                
                if tool_name in self.tools:
                    try:
                        result = self.tools[tool_name](**tool_input)
                    except Exception as e:
                        result = f"Error: {e}"
                    
                    iteration["type"] = "tool"
                    iteration["tool"] = tool_name
                    iteration["result"] = result
                    messages.append(Message.assistant(content))
                    messages.append(Message.user(f"Result: {result}"))
                else:
                    messages.append(Message.assistant(content))
                    messages.append(Message.user(f"Unknown tool: {tool_name}"))
            else:
                messages.append(Message.assistant(content))
                messages.append(Message.user("Continue or give FINAL answer."))
            
            iterations.append(iteration)
        
        return {"answer": None, "error": "Max iterations", "iterations": iterations}
    
    async def arun(self, query: str) -> Dict[str, Any]:
        """Run asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.run(query))


# =============================================================================
# RAG - Retrieval Augmented Generation
# =============================================================================

class RAG:
    """
    Simple RAG with keyword matching.
    Plug in your own retriever for production.
    
    Usage:
        rag = RAG(llm)
        rag.add([Document("JARVIS was created by Kent.")])
        result = rag.query("Who created JARVIS?")
    """
    
    def __init__(self, llm: LLM, top_k: int = 3):
        self.llm = llm
        self.top_k = top_k
        self.documents: List[Document] = []
    
    def add(self, docs: List[Document]):
        """Add documents."""
        self.documents.extend(docs)
    
    def clear(self):
        """Clear all documents."""
        self.documents = []
    
    def _retrieve(self, query: str) -> List[Document]:
        """Simple keyword retrieval."""
        query_words = set(query.lower().split())
        scored = []
        
        for doc in self.documents:
            doc_words = set(doc.content.lower().split())
            score = len(query_words & doc_words)
            if score > 0:
                scored.append((score, doc))
        
        scored.sort(key=lambda x: -x[0])
        return [doc for _, doc in scored[:self.top_k]]
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query synchronously."""
        docs = self._retrieve(question)
        
        if not docs:
            context = "No relevant documents found."
        else:
            context = "\n\n---\n\n".join(d.content for d in docs)
        
        messages = [
            Message.system("Answer based on context. Say if not found."),
            Message.user(f"Context:\n{context}\n\nQuestion: {question}")
        ]
        
        response = self.llm.complete(messages)
        
        return {
            "answer": response.content,
            "sources": [{"id": d.id, **d.metadata} for d in docs],
            "tokens": response.total_tokens
        }
    
    async def aquery(self, question: str) -> Dict[str, Any]:
        """Query asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.query(question))


# =============================================================================
# MEMORY - Conversation memory
# =============================================================================

class Memory:
    """
    Conversation memory with configurable window.
    
    Usage:
        memory = Memory(max_messages=20)
        memory.add("user", "Hello")
        memory.add("assistant", "Hi!")
    """
    
    def __init__(self, max_messages: int = 50):
        self.max_messages = max_messages
        self.messages: List[Message] = []
    
    def add(self, role: str, content: str) -> "Memory":
        """Add a message."""
        self.messages.append(Message(role, content))
        self._trim()
        return self
    
    def _trim(self):
        """Trim to max messages, keeping system."""
        if len(self.messages) <= self.max_messages:
            return
        
        system = [m for m in self.messages if m.role == "system"]
        others = [m for m in self.messages if m.role != "system"]
        keep = self.max_messages - len(system)
        self.messages = system + others[-keep:]
    
    def get(self) -> List[Message]:
        """Get all messages."""
        return self.messages.copy()
    
    def clear(self):
        """Clear non-system messages."""
        self.messages = [m for m in self.messages if m.role == "system"]


class Conversation:
    """
    Conversational chain with memory.
    
    Usage:
        conv = Conversation(llm, system="You are a pirate.")
        print(conv.chat("Hello!"))  # "Ahoy!"
    """
    
    def __init__(
        self,
        llm: LLM,
        system: str = "You are a helpful assistant.",
        memory: Memory = None
    ):
        self.llm = llm
        self.memory = memory or Memory()
        self.memory.add("system", system)
    
    def chat(self, message: str) -> str:
        """Chat synchronously."""
        self.memory.add("user", message)
        response = self.llm.complete(self.memory.get())
        self.memory.add("assistant", response.content)
        return response.content
    
    async def achat(self, message: str) -> str:
        """Chat asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.chat(message))
    
    def reset(self):
        """Reset conversation."""
        self.memory.clear()


# =============================================================================
# PARALLEL - Run multiple completions in parallel
# =============================================================================

class Parallel:
    """
    Run multiple LLM calls in parallel.
    
    Usage:
        results = Parallel.run([
            (llm1, "Question 1"),
            (llm2, "Question 2"),
        ])
    """
    
    @staticmethod
    def run(tasks: List[tuple], max_workers: int = 5) -> List[Response]:
        """Run tasks in parallel using threads."""
        def execute(task):
            llm, prompt = task[:2]
            kwargs = task[2] if len(task) > 2 else {}
            return llm.complete([Message.user(prompt)], **kwargs)
        
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(execute, tasks))
    
    @staticmethod
    async def arun(tasks: List[tuple]) -> List[Response]:
        """Run tasks in parallel using asyncio."""
        async def execute(task):
            llm, prompt = task[:2]
            kwargs = task[2] if len(task) > 2 else {}
            return await llm.acomplete([Message.user(prompt)], **kwargs)
        
        return await asyncio.gather(*[execute(t) for t in tasks])


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def complete(prompt: str, provider: str = "anthropic", **kwargs) -> str:
    """
    Quick completion.
    
    Usage:
        response = complete("What is 2+2?")
    """
    providers = {
        "anthropic": Anthropic,
        "openai": OpenAI,
        "ollama": Ollama,
        "groq": Groq,
        "mistral": Mistral,
        "deepseek": DeepSeek,
    }
    
    llm_class = providers.get(provider)
    if not llm_class:
        raise ValueError(f"Unknown provider: {provider}")
    
    llm = llm_class(**{k: v for k, v in kwargs.items() if k in ['api_key', 'model', 'base_url']})
    return llm(prompt)


async def acomplete(prompt: str, provider: str = "anthropic", **kwargs) -> str:
    """Async quick completion."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: complete(prompt, provider, **kwargs)
    )


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI entry point."""
    import sys
    
    print("=" * 60)
    print("SimpleChain - Zero Dependency LLM Framework")
    print("=" * 60)
    print(f"\nVersion: {__version__}")
    print(f"Author: {__author__}")
    print(f"License: {__license__}")
    print("\nDependencies: ZERO (pure Python stdlib)")
    print("\nSupported Providers:")
    print("  - Anthropic Claude")
    print("  - OpenAI GPT")
    print("  - Groq")
    print("  - Mistral")
    print("  - DeepSeek")
    print("  - Ollama (local)")
    print("\nFeatures:")
    print("  - LLM adapters for all major providers")
    print("  - Chain: Sequential LLM calls")
    print("  - Router: Route to different LLMs")
    print("  - Agent: Tool-using agent")
    print("  - RAG: Document Q&A")
    print("  - Memory: Conversation history")
    print("  - Parallel: Concurrent calls")
    print("\nUsage:")
    print("  from simplechain import Anthropic, Chain, Agent")
    print("  llm = Anthropic()")
    print("  print(llm('Hello!'))")
    print("\nGitHub: https://github.com/KentStone/simplechain")
    
    # Live demo if API key available
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("\n" + "-" * 60)
        print("Live Demo:")
        try:
            llm = Anthropic()
            response = llm("What is 2+2? Answer in one word.")
            print(f"  Q: What is 2+2?")
            print(f"  A: {response}")
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    main()
