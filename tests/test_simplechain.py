"""
Tests for SimpleChain
"""

import os
import pytest
from unittest.mock import patch, MagicMock

# Import from parent
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stonechain import (
    Message,
    Response,
    Tool,
    Document,
    Provider,
    Chain,
    Router,
    Agent,
    RAG,
    Memory,
    Conversation,
    Parallel,
    HTTP,
    Step,
)


class TestMessage:
    """Test Message class."""
    
    def test_create_message(self):
        msg = Message("user", "Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
    
    def test_to_dict(self):
        msg = Message("user", "Hello")
        d = msg.to_dict()
        assert d == {"role": "user", "content": "Hello"}
    
    def test_system_helper(self):
        msg = Message.system("You are helpful.")
        assert msg.role == "system"
        assert msg.content == "You are helpful."
    
    def test_user_helper(self):
        msg = Message.user("Hello!")
        assert msg.role == "user"
    
    def test_assistant_helper(self):
        msg = Message.assistant("Hi there!")
        assert msg.role == "assistant"


class TestResponse:
    """Test Response class."""
    
    def test_total_tokens(self):
        resp = Response(
            content="Hello",
            model="test",
            provider=Provider.ANTHROPIC,
            input_tokens=10,
            output_tokens=5
        )
        assert resp.total_tokens == 15
    
    def test_tokens_per_second(self):
        resp = Response(
            content="Hello",
            model="test",
            provider=Provider.ANTHROPIC,
            output_tokens=100,
            latency_ms=1000  # 1 second
        )
        assert resp.tokens_per_second == 100.0


class TestTool:
    """Test Tool class."""
    
    def test_create_tool(self):
        def calc(x: int) -> int:
            return x * 2
        
        tool = Tool(
            name="calculator",
            description="Double a number",
            parameters={"x": {"type": "integer"}},
            function=calc
        )
        
        assert tool.name == "calculator"
        assert tool(x=5) == 10
    
    def test_openai_schema(self):
        tool = Tool(
            name="test",
            description="Test tool",
            parameters={"x": {"type": "string"}},
            function=lambda x: x
        )
        
        schema = tool.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "test"
    
    def test_anthropic_schema(self):
        tool = Tool(
            name="test",
            description="Test tool",
            parameters={"x": {"type": "string"}},
            function=lambda x: x
        )
        
        schema = tool.to_anthropic_schema()
        assert schema["name"] == "test"
        assert "input_schema" in schema


class TestDocument:
    """Test Document class."""
    
    def test_create_document(self):
        doc = Document("Hello world")
        assert doc.content == "Hello world"
        assert doc.id is not None  # Auto-generated
    
    def test_document_with_metadata(self):
        doc = Document("Hello", metadata={"source": "test"})
        assert doc.metadata["source"] == "test"
    
    def test_document_with_id(self):
        doc = Document("Hello", id="custom-id")
        assert doc.id == "custom-id"


class TestMemory:
    """Test Memory class."""
    
    def test_add_message(self):
        memory = Memory()
        memory.add("user", "Hello")
        memory.add("assistant", "Hi!")
        
        messages = memory.get()
        assert len(messages) == 2
    
    def test_max_messages(self):
        memory = Memory(max_messages=3)
        memory.add("system", "You are helpful.")
        memory.add("user", "1")
        memory.add("assistant", "1")
        memory.add("user", "2")
        memory.add("assistant", "2")
        
        messages = memory.get()
        # Should keep system + last 2 pairs
        assert len(messages) <= 3
        # System should always be kept
        assert any(m.role == "system" for m in messages)
    
    def test_clear(self):
        memory = Memory()
        memory.add("system", "System")
        memory.add("user", "Hello")
        memory.clear()
        
        messages = memory.get()
        assert len(messages) == 1  # Only system kept
        assert messages[0].role == "system"


class TestHTTP:
    """Test HTTP client."""
    
    def test_singleton(self):
        h1 = HTTP()
        h2 = HTTP()
        assert h1 is h2


class TestChain:
    """Test Chain class."""
    
    def test_chain_creation(self):
        # Mock LLM
        mock_llm = MagicMock()
        mock_llm.complete.return_value = Response(
            content="Test output",
            model="test",
            provider=Provider.ANTHROPIC
        )
        
        chain = Chain(mock_llm)
        chain.add("step1", "Process: {input}", "output1")
        
        assert len(chain.steps) == 1
        assert chain.steps[0].name == "step1"
    
    def test_step_format(self):
        step = Step("test", "Hello {name}!", "out")
        result = step.format({"name": "World"})
        assert result == "Hello World!"


class TestRouter:
    """Test Router class."""
    
    def test_router_creation(self):
        mock_llm = MagicMock()
        router = Router(default=mock_llm)
        
        assert router.default is mock_llm
    
    def test_add_route(self):
        mock_llm = MagicMock()
        router = Router()
        router.add("test", lambda x: True, mock_llm)
        
        assert len(router.routes) == 1


class TestRAG:
    """Test RAG class."""
    
    def test_add_documents(self):
        mock_llm = MagicMock()
        rag = RAG(mock_llm)
        
        rag.add([
            Document("Doc 1"),
            Document("Doc 2"),
        ])
        
        assert len(rag.documents) == 2
    
    def test_clear_documents(self):
        mock_llm = MagicMock()
        rag = RAG(mock_llm)
        
        rag.add([Document("Doc 1")])
        rag.clear()
        
        assert len(rag.documents) == 0
    
    def test_retrieve(self):
        mock_llm = MagicMock()
        rag = RAG(mock_llm, top_k=2)
        
        rag.add([
            Document("The cat sat on the mat"),
            Document("Dogs are great pets"),
            Document("The cat is fluffy"),
        ])
        
        # Simple keyword retrieval
        docs = rag._retrieve("cat")
        assert len(docs) <= 2
        assert any("cat" in d.content.lower() for d in docs)


class TestAgent:
    """Test Agent class."""
    
    def test_agent_creation(self):
        mock_llm = MagicMock()
        tools = [
            Tool("test", "Test", {}, lambda: "result")
        ]
        
        agent = Agent(mock_llm, tools)
        assert "test" in agent.tools


class TestParallel:
    """Test Parallel execution."""
    
    def test_parallel_run(self):
        mock_llm = MagicMock()
        mock_llm.complete.return_value = Response(
            content="Response",
            model="test",
            provider=Provider.ANTHROPIC
        )
        
        results = Parallel.run([
            (mock_llm, "Query 1"),
            (mock_llm, "Query 2"),
        ])
        
        assert len(results) == 2
        assert mock_llm.complete.call_count == 2


# Integration tests (require API keys)
@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set"
)
class TestAnthropicIntegration:
    """Integration tests with real Anthropic API."""
    
    def test_simple_completion(self):
        from simplechain import Anthropic
        
        llm = Anthropic()
        response = llm("What is 2+2? Answer with just the number.")
        
        assert "4" in response


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)
class TestOpenAIIntegration:
    """Integration tests with real OpenAI API."""
    
    def test_simple_completion(self):
        from simplechain import OpenAI
        
        llm = OpenAI()
        response = llm("What is 2+2? Answer with just the number.")
        
        assert "4" in response


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
