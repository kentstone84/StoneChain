# Advanced Usage

## Custom Providers

SimpleChain makes it easy to add custom LLM providers.

### OpenAI-Compatible APIs

Many providers use OpenAI-compatible APIs. Just change the base URL:

```python
from simplechain import OpenAI

# Together AI
together = OpenAI(
    api_key="your-together-key",
    base_url="https://api.together.xyz/v1",
    model="meta-llama/Llama-3-70b-chat-hf"
)

# Anyscale
anyscale = OpenAI(
    api_key="your-anyscale-key",
    base_url="https://api.endpoints.anyscale.com/v1",
    model="meta-llama/Llama-3-70b-chat-hf"
)

# Fireworks
fireworks = OpenAI(
    api_key="your-fireworks-key",
    base_url="https://api.fireworks.ai/inference/v1",
    model="accounts/fireworks/models/llama-v3-70b-instruct"
)

# Local vLLM
vllm = OpenAI(
    api_key="not-needed",
    base_url="http://localhost:8000/v1",
    model="meta-llama/Llama-3-70b"
)
```

### Custom Adapter

For non-OpenAI-compatible APIs:

```python
from simplechain import LLM, Message, Response, Provider, http
import time

class CustomLLM(LLM):
    provider = Provider.CUSTOM
    
    def __init__(self, api_key: str, model: str = "default"):
        self.api_key = api_key
        self.default_model = model
        self.base_url = "https://api.custom-llm.com/v1"
    
    def complete(
        self,
        messages: list,
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> Response:
        start = time.perf_counter()
        
        # Build your API payload
        payload = {
            "model": model or self.default_model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Make request
        data = http.post(
            f"{self.base_url}/chat",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            data=payload
        )
        
        latency = (time.perf_counter() - start) * 1000
        
        # Parse response (adjust for your API)
        return Response(
            content=data["choices"][0]["message"]["content"],
            model=model or self.default_model,
            provider=self.provider,
            input_tokens=data.get("usage", {}).get("input_tokens", 0),
            output_tokens=data.get("usage", {}).get("output_tokens", 0),
            latency_ms=latency,
            raw=data
        )

# Use it
llm = CustomLLM(api_key="your-key")
print(llm("Hello!"))
```

## Advanced Chains

### Conditional Steps

```python
from simplechain import Anthropic, Chain

llm = Anthropic()

def maybe_translate(context):
    """Only translate if not English."""
    if "non-english" in context.get("analysis", "").lower():
        return True
    return False

chain = Chain(llm)
chain.add("analyze", "Analyze the language of: {input}", "analysis")
chain.add("translate", "Translate to English: {input}", "translated")  # Always runs
chain.add("summarize", "Summarize: {translated}", "summary")

result = chain.run(input="Bonjour le monde!")
```

### Branching Chains

```python
from simplechain import Anthropic, Chain, Router

llm = Anthropic()

# Technical chain
tech_chain = Chain(llm, system="You are a technical expert.")
tech_chain.add("explain", "Explain technically: {input}", "explanation")
tech_chain.add("example", "Give code example for: {explanation}", "code")

# Simple chain
simple_chain = Chain(llm, system="Explain like I'm 5.")
simple_chain.add("explain", "Explain simply: {input}", "explanation")

# Route based on complexity
router = Router()
router.add("technical", lambda x: "technical" in x.lower() or "code" in x.lower(), tech_chain)
router.add("simple", lambda x: "simple" in x.lower() or "eli5" in x.lower(), simple_chain)

result = router.route("Explain recursion technically")
```

## Advanced Agents

### Multi-Tool Agent

```python
from simplechain import Anthropic, Agent, Tool
import json

def calculator(expression: str) -> str:
    try:
        return str(eval(expression))
    except:
        return "Error"

def web_search(query: str) -> str:
    # Implement your search
    return f"Results for: {query}"

def file_read(path: str) -> str:
    try:
        with open(path) as f:
            return f.read()[:1000]
    except:
        return "File not found"

def file_write(path: str, content: str) -> str:
    try:
        with open(path, 'w') as f:
            f.write(content)
        return "Written successfully"
    except Exception as e:
        return f"Error: {e}"

tools = [
    Tool("calculator", "Math calculations", 
         {"expression": {"type": "string"}}, calculator),
    Tool("search", "Search the web",
         {"query": {"type": "string"}}, web_search),
    Tool("read_file", "Read a file",
         {"path": {"type": "string"}}, file_read),
    Tool("write_file", "Write to a file",
         {"path": {"type": "string"}, "content": {"type": "string"}}, file_write),
]

agent = Agent(Anthropic(), tools, max_iterations=15)
result = agent.run("Calculate 15*23, then write the result to result.txt")
```

### Custom Agent System Prompt

```python
from simplechain import Anthropic, Agent, Tool

tools = [...]

custom_system = """You are a helpful coding assistant.

Available tools:
{tools}

When using a tool, format as:
TOOL: tool_name
INPUT: {{"param": "value"}}

When done, say:
FINAL: your answer

Think step by step. Always verify your work.
"""

agent = Agent(
    Anthropic(),
    tools,
    system=custom_system.format(
        tools="\n".join(f"- {t.name}: {t.description}" for t in tools)
    )
)
```

## Advanced RAG

### Custom Retriever

```python
from simplechain import Anthropic, RAG, Document

class CustomRAG(RAG):
    """RAG with custom retrieval logic."""
    
    def _retrieve(self, query: str) -> list:
        # Implement your own retrieval
        # Could use embeddings, BM25, hybrid, etc.
        
        # Example: TF-IDF-like scoring
        query_terms = set(query.lower().split())
        scored = []
        
        for doc in self.documents:
            doc_terms = set(doc.content.lower().split())
            
            # Calculate simple TF-IDF-like score
            overlap = query_terms & doc_terms
            if overlap:
                tf = len(overlap) / len(doc_terms)
                idf = len(self.documents) / sum(
                    1 for d in self.documents 
                    if any(t in d.content.lower() for t in overlap)
                )
                score = tf * idf
                scored.append((score, doc))
        
        scored.sort(key=lambda x: -x[0])
        return [doc for _, doc in scored[:self.top_k]]

rag = CustomRAG(Anthropic())
rag.add([Document("..."), Document("...")])
result = rag.query("Your question")
```

### RAG with Metadata Filtering

```python
from simplechain import Anthropic, RAG, Document

class FilteredRAG(RAG):
    def query(self, question: str, filter_metadata: dict = None) -> dict:
        # Filter documents first
        if filter_metadata:
            filtered_docs = [
                d for d in self.documents
                if all(d.metadata.get(k) == v for k, v in filter_metadata.items())
            ]
            original_docs = self.documents
            self.documents = filtered_docs
        
        result = super().query(question)
        
        if filter_metadata:
            self.documents = original_docs
        
        return result

rag = FilteredRAG(Anthropic())
rag.add([
    Document("Python guide", metadata={"language": "python", "type": "guide"}),
    Document("Python API ref", metadata={"language": "python", "type": "api"}),
    Document("JavaScript guide", metadata={"language": "js", "type": "guide"}),
])

# Only search Python guides
result = rag.query("How to use decorators?", filter_metadata={"language": "python"})
```

## Streaming (Coming Soon)

```python
# Planned for v1.1
from simplechain import Anthropic

llm = Anthropic()

for chunk in llm.stream("Tell me a story"):
    print(chunk, end="", flush=True)
```

## Error Handling

```python
from simplechain import Anthropic, APIError, ConfigError

try:
    llm = Anthropic()
    response = llm("Hello")
except ConfigError as e:
    print(f"Configuration error: {e}")
except APIError as e:
    print(f"API error (status {e.status_code}): {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Retry Logic

```python
from simplechain import Anthropic, APIError
import time

def complete_with_retry(llm, prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            return llm(prompt)
        except APIError as e:
            if e.status_code == 429:  # Rate limit
                wait = 2 ** attempt
                print(f"Rate limited, waiting {wait}s...")
                time.sleep(wait)
            elif e.status_code >= 500:  # Server error
                wait = 2 ** attempt
                print(f"Server error, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise Exception("Max retries exceeded")

llm = Anthropic()
response = complete_with_retry(llm, "Hello!")
```

## Logging

```python
from simplechain import Anthropic, Message
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("simplechain")

class LoggingLLM(Anthropic):
    def complete(self, messages, **kwargs):
        logger.info(f"Request: {len(messages)} messages")
        response = super().complete(messages, **kwargs)
        logger.info(f"Response: {response.total_tokens} tokens, {response.latency_ms:.0f}ms")
        return response

llm = LoggingLLM()
llm("Hello!")
```
