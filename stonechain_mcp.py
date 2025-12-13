#!/usr/bin/env python3
"""
StoneChain MCP Support
======================

Model Context Protocol (MCP) client support for StoneChain.
Zero dependencies. Pure stdlib. Built like a rock.

Usage:
    from stonechain import Anthropic
    from stonechain_mcp import MCPClient, StdioTransport, HTTPTransport
    
    # Connect to MCP servers
    client = MCPClient({
        "math": StdioTransport("python", ["math_server.py"]),
        "weather": HTTPTransport("http://localhost:8000/mcp"),
    })
    
    # Get tools and use with StoneChain
    tools = await client.get_tools()
    agent = Agent(Anthropic(), tools)

Author: Kent Stone
License: MIT
"""

__version__ = "1.0.0"

import json
import subprocess
import asyncio
import urllib.request
import urllib.error
import ssl
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Union
from abc import ABC, abstractmethod


# =============================================================================
# TRANSPORTS
# =============================================================================

class Transport(ABC):
    """Base transport for MCP communication."""
    
    @abstractmethod
    async def send(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send a JSON-RPC request and return response."""
        pass
    
    @abstractmethod
    async def connect(self):
        """Establish connection."""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Close connection."""
        pass


class StdioTransport(Transport):
    """
    Stdio transport - launches server as subprocess.
    
    Usage:
        transport = StdioTransport("python", ["math_server.py"])
    """
    
    def __init__(self, command: str, args: List[str] = None, env: Dict[str, str] = None):
        self.command = command
        self.args = args or []
        self.env = env
        self.process: Optional[subprocess.Popen] = None
        self._request_id = 0
    
    async def connect(self):
        """Start the subprocess."""
        env = os.environ.copy()
        if self.env:
            env.update(self.env)
        
        self.process = subprocess.Popen(
            [self.command] + self.args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=1
        )
    
    async def disconnect(self):
        """Terminate the subprocess."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
    
    async def send(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send JSON-RPC request via stdio."""
        if not self.process:
            await self.connect()
        
        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
        }
        if params:
            request["params"] = params
        
        # Write request
        request_line = json.dumps(request) + "\n"
        self.process.stdin.write(request_line)
        self.process.stdin.flush()
        
        # Read response (run in thread to not block)
        loop = asyncio.get_event_loop()
        response_line = await loop.run_in_executor(None, self.process.stdout.readline)
        
        if not response_line:
            raise ConnectionError("No response from MCP server")
        
        response = json.loads(response_line)
        
        if "error" in response:
            raise MCPError(response["error"].get("message", "Unknown error"), response["error"].get("code"))
        
        return response.get("result", {})


class HTTPTransport(Transport):
    """
    HTTP transport for remote MCP servers.
    
    Usage:
        transport = HTTPTransport("http://localhost:8000/mcp")
    """
    
    def __init__(self, url: str, headers: Dict[str, str] = None, auth: Any = None):
        self.url = url
        self.headers = headers or {}
        self.auth = auth
        self._ssl_context = ssl.create_default_context()
        self._request_id = 0
        self._session_id: Optional[str] = None
    
    async def connect(self):
        """Initialize HTTP connection (get session if needed)."""
        # For stateless HTTP, just validate the endpoint
        pass
    
    async def disconnect(self):
        """Close HTTP session."""
        self._session_id = None
    
    async def send(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send JSON-RPC request via HTTP POST."""
        self._request_id += 1
        
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
        }
        if params:
            request["params"] = params
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            **self.headers
        }
        
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        
        json_data = json.dumps(request).encode('utf-8')
        req = urllib.request.Request(self.url, data=json_data, headers=headers, method='POST')
        
        loop = asyncio.get_event_loop()
        
        def do_request():
            try:
                with urllib.request.urlopen(req, timeout=120, context=self._ssl_context) as resp:
                    # Capture session ID if returned
                    session_id = resp.headers.get("Mcp-Session-Id")
                    if session_id:
                        self._session_id = session_id
                    return json.loads(resp.read().decode('utf-8'))
            except urllib.error.HTTPError as e:
                body = e.read().decode('utf-8')
                raise MCPError(f"HTTP {e.code}: {body}", e.code)
            except urllib.error.URLError as e:
                raise ConnectionError(f"Connection failed: {e.reason}")
        
        response = await loop.run_in_executor(None, do_request)
        
        if "error" in response:
            raise MCPError(response["error"].get("message", "Unknown error"), response["error"].get("code"))
        
        return response.get("result", {})


# =============================================================================
# MCP CLIENT
# =============================================================================

class MCPError(Exception):
    """MCP protocol error."""
    def __init__(self, message: str, code: int = None):
        super().__init__(message)
        self.code = code


@dataclass
class MCPTool:
    """Tool definition from MCP server."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    server_name: str
    
    def to_stonechain_tool(self, client: "MCPClient") -> "Tool":
        """Convert to StoneChain Tool."""
        from stonechain import Tool
        
        # Extract parameters from JSON schema
        properties = self.input_schema.get("properties", {})
        parameters = {}
        for key, schema in properties.items():
            parameters[key] = {
                "type": schema.get("type", "string"),
                "description": schema.get("description", "")
            }
        
        # Create wrapper function that calls MCP
        server_name = self.server_name
        tool_name = self.name
        
        async def call_tool(**kwargs):
            return await client.call_tool(server_name, tool_name, kwargs)
        
        # Sync wrapper
        def sync_call_tool(**kwargs):
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(call_tool(**kwargs))
        
        return Tool(
            name=f"{server_name}_{self.name}" if client.prefix_tool_names else self.name,
            description=self.description,
            parameters=parameters,
            function=sync_call_tool
        )


@dataclass
class MCPResource:
    """Resource from MCP server."""
    uri: str
    name: str
    description: Optional[str] = None
    mime_type: Optional[str] = None


@dataclass
class MCPPrompt:
    """Prompt template from MCP server."""
    name: str
    description: Optional[str] = None
    arguments: List[Dict[str, Any]] = field(default_factory=list)


class MCPClient:
    """
    Multi-server MCP client for StoneChain.
    
    Usage:
        client = MCPClient({
            "math": StdioTransport("python", ["math_server.py"]),
            "weather": HTTPTransport("http://localhost:8000/mcp"),
        })
        
        tools = await client.get_tools()
        agent = Agent(llm, tools)
    """
    
    def __init__(
        self,
        servers: Dict[str, Transport],
        prefix_tool_names: bool = True,
        tool_interceptors: List[Callable] = None
    ):
        self.servers = servers
        self.prefix_tool_names = prefix_tool_names
        self.tool_interceptors = tool_interceptors or []
        self._initialized: Dict[str, bool] = {}
        self._tools_cache: Dict[str, List[MCPTool]] = {}
    
    async def _ensure_initialized(self, server_name: str):
        """Initialize server connection if needed."""
        if server_name not in self._initialized:
            transport = self.servers[server_name]
            await transport.connect()
            
            # Send initialize request
            result = await transport.send("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                    "resources": {},
                    "prompts": {}
                },
                "clientInfo": {
                    "name": "stonechain",
                    "version": __version__
                }
            })
            
            # Send initialized notification
            await transport.send("notifications/initialized", {})
            
            self._initialized[server_name] = True
    
    async def get_tools(self, server_name: str = None) -> List["Tool"]:
        """
        Get tools from MCP server(s) as StoneChain Tools.
        
        Args:
            server_name: Specific server, or None for all servers
        
        Returns:
            List of StoneChain Tool objects
        """
        from stonechain import Tool
        
        tools = []
        servers = [server_name] if server_name else list(self.servers.keys())
        
        for name in servers:
            await self._ensure_initialized(name)
            transport = self.servers[name]
            
            # List tools
            result = await transport.send("tools/list", {})
            
            mcp_tools = []
            for tool_def in result.get("tools", []):
                mcp_tool = MCPTool(
                    name=tool_def["name"],
                    description=tool_def.get("description", ""),
                    input_schema=tool_def.get("inputSchema", {}),
                    server_name=name
                )
                mcp_tools.append(mcp_tool)
                tools.append(mcp_tool.to_stonechain_tool(self))
            
            self._tools_cache[name] = mcp_tools
        
        return tools
    
    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a tool on an MCP server.
        
        Args:
            server_name: Name of the server
            tool_name: Name of the tool
            arguments: Tool arguments
        
        Returns:
            Tool result
        """
        await self._ensure_initialized(server_name)
        transport = self.servers[server_name]
        
        # Apply interceptors
        request = {"server": server_name, "tool": tool_name, "args": arguments}
        
        for interceptor in self.tool_interceptors:
            request = await interceptor(request)
            if request is None:
                return None
        
        result = await transport.send("tools/call", {
            "name": request["tool"],
            "arguments": request["args"]
        })
        
        # Extract content from result
        content_parts = result.get("content", [])
        if not content_parts:
            return None
        
        # Handle different content types
        texts = []
        for part in content_parts:
            if part.get("type") == "text":
                texts.append(part.get("text", ""))
            elif part.get("type") == "image":
                texts.append(f"[Image: {part.get('mimeType', 'unknown')}]")
        
        return "\n".join(texts) if texts else str(result)
    
    async def get_resources(self, server_name: str, uris: List[str] = None) -> List[Dict[str, Any]]:
        """
        Get resources from an MCP server.
        
        Args:
            server_name: Name of the server
            uris: Specific URIs to fetch, or None for all
        
        Returns:
            List of resource contents
        """
        await self._ensure_initialized(server_name)
        transport = self.servers[server_name]
        
        # List resources
        result = await transport.send("resources/list", {})
        resources = result.get("resources", [])
        
        if uris:
            resources = [r for r in resources if r.get("uri") in uris]
        
        # Read each resource
        contents = []
        for resource in resources:
            read_result = await transport.send("resources/read", {"uri": resource["uri"]})
            for content in read_result.get("contents", []):
                contents.append({
                    "uri": resource["uri"],
                    "name": resource.get("name", ""),
                    "mime_type": content.get("mimeType"),
                    "text": content.get("text"),
                    "blob": content.get("blob")
                })
        
        return contents
    
    async def get_prompt(self, server_name: str, prompt_name: str, arguments: Dict[str, str] = None) -> List[Dict[str, Any]]:
        """
        Get a prompt from an MCP server.
        
        Args:
            server_name: Name of the server
            prompt_name: Name of the prompt
            arguments: Prompt arguments
        
        Returns:
            List of messages from the prompt
        """
        await self._ensure_initialized(server_name)
        transport = self.servers[server_name]
        
        params = {"name": prompt_name}
        if arguments:
            params["arguments"] = arguments
        
        result = await transport.send("prompts/get", params)
        
        messages = []
        for msg in result.get("messages", []):
            content = msg.get("content", {})
            messages.append({
                "role": msg.get("role", "user"),
                "content": content.get("text", "") if isinstance(content, dict) else str(content)
            })
        
        return messages
    
    async def close(self):
        """Close all server connections."""
        for name, transport in self.servers.items():
            try:
                await transport.disconnect()
            except:
                pass
        self._initialized.clear()
        self._tools_cache.clear()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, *args):
        await self.close()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def stdio_server(command: str, *args, env: Dict[str, str] = None) -> StdioTransport:
    """Create a stdio transport for a local MCP server."""
    return StdioTransport(command, list(args), env)


def http_server(url: str, headers: Dict[str, str] = None) -> HTTPTransport:
    """Create an HTTP transport for a remote MCP server."""
    return HTTPTransport(url, headers)


if __name__ == "__main__":
    print("StoneChain MCP Client")
    print("=" * 40)
    print(f"Version: {__version__}")
    print("\nFeatures:")
    print("  - Stdio transport (local servers)")
    print("  - HTTP transport (remote servers)")
    print("  - Multi-server support")
    print("  - Tool prefix to avoid conflicts")
    print("  - Tool interceptors")
    print("  - Resource fetching")
    print("  - Prompt templates")
    print("\nUsage:")
    print("  from stonechain_mcp import MCPClient, StdioTransport, HTTPTransport")
    print("  client = MCPClient({'math': StdioTransport('python', ['server.py'])})")
    print("  tools = await client.get_tools()")
