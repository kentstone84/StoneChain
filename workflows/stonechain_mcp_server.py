#!/usr/bin/env python3
"""
StoneChain MCP Server
=====================

Build MCP servers with zero dependencies. FastMCP alternative.

Usage:
    from stonechain_mcp_server import MCPServer
    
    server = MCPServer("Math")
    
    @server.tool()
    def add(a: int, b: int) -> int:
        '''Add two numbers'''
        return a + b
    
    server.run()

Author: Kent Stone
License: MIT
"""

__version__ = "1.0.0"

import json
import sys
import inspect
import asyncio
from typing import Any, Callable, Dict, List, Optional, get_type_hints
from dataclasses import dataclass, field


# =============================================================================
# TYPE MAPPING
# =============================================================================

PYTHON_TO_JSON_TYPE = {
    int: "integer",
    float: "number",
    str: "string",
    bool: "boolean",
    list: "array",
    dict: "object",
    type(None): "null",
}


def python_type_to_json_schema(py_type) -> Dict[str, Any]:
    """Convert Python type hint to JSON Schema."""
    if py_type in PYTHON_TO_JSON_TYPE:
        return {"type": PYTHON_TO_JSON_TYPE[py_type]}
    
    # Handle Optional
    origin = getattr(py_type, "__origin__", None)
    if origin is type(None):
        return {"type": "null"}
    
    # Handle List[X]
    if origin is list:
        args = getattr(py_type, "__args__", (Any,))
        return {
            "type": "array",
            "items": python_type_to_json_schema(args[0]) if args else {}
        }
    
    # Handle Dict[K, V]
    if origin is dict:
        return {"type": "object"}
    
    # Default to string
    return {"type": "string"}


# =============================================================================
# MCP SERVER
# =============================================================================

@dataclass
class ToolDef:
    """Tool definition."""
    name: str
    description: str
    function: Callable
    parameters: Dict[str, Any]


@dataclass 
class ResourceDef:
    """Resource definition."""
    uri: str
    name: str
    description: str
    mime_type: str
    handler: Callable


@dataclass
class PromptDef:
    """Prompt definition."""
    name: str
    description: str
    arguments: List[Dict[str, Any]]
    handler: Callable


class MCPServer:
    """
    MCP Server builder. Zero dependencies.
    
    Usage:
        server = MCPServer("Math")
        
        @server.tool()
        def add(a: int, b: int) -> int:
            '''Add two numbers'''
            return a + b
        
        @server.tool()
        def multiply(a: int, b: int) -> int:
            '''Multiply two numbers'''
            return a * b
        
        server.run()
    """
    
    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.tools: Dict[str, ToolDef] = {}
        self.resources: Dict[str, ResourceDef] = {}
        self.prompts: Dict[str, PromptDef] = {}
    
    def tool(self, name: str = None, description: str = None):
        """
        Decorator to register a tool.
        
        @server.tool()
        def add(a: int, b: int) -> int:
            '''Add two numbers'''
            return a + b
        """
        def decorator(func: Callable):
            tool_name = name or func.__name__
            tool_desc = description or (func.__doc__ or "").strip()
            
            # Extract parameters from type hints
            hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}
            sig = inspect.signature(func)
            
            properties = {}
            required = []
            
            for param_name, param in sig.parameters.items():
                if param_name in ("self", "ctx", "context"):
                    continue
                
                param_type = hints.get(param_name, str)
                param_schema = python_type_to_json_schema(param_type)
                
                # Add description from docstring if available
                properties[param_name] = param_schema
                
                if param.default is inspect.Parameter.empty:
                    required.append(param_name)
            
            input_schema = {
                "type": "object",
                "properties": properties,
                "required": required
            }
            
            self.tools[tool_name] = ToolDef(
                name=tool_name,
                description=tool_desc,
                function=func,
                parameters=input_schema
            )
            
            return func
        
        return decorator
    
    def resource(self, uri: str, name: str = None, description: str = "", mime_type: str = "text/plain"):
        """
        Decorator to register a resource.
        
        @server.resource("file:///config.json", name="Config")
        def get_config():
            return json.dumps({"key": "value"})
        """
        def decorator(func: Callable):
            resource_name = name or func.__name__
            self.resources[uri] = ResourceDef(
                uri=uri,
                name=resource_name,
                description=description or (func.__doc__ or "").strip(),
                mime_type=mime_type,
                handler=func
            )
            return func
        return decorator
    
    def prompt(self, name: str = None, description: str = None):
        """
        Decorator to register a prompt template.
        
        @server.prompt()
        def summarize(text: str):
            return f"Please summarize the following text:\\n\\n{text}"
        """
        def decorator(func: Callable):
            prompt_name = name or func.__name__
            prompt_desc = description or (func.__doc__ or "").strip()
            
            # Extract arguments from signature
            sig = inspect.signature(func)
            hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}
            
            arguments = []
            for param_name, param in sig.parameters.items():
                if param_name in ("self", "ctx", "context"):
                    continue
                
                arguments.append({
                    "name": param_name,
                    "required": param.default is inspect.Parameter.empty
                })
            
            self.prompts[prompt_name] = PromptDef(
                name=prompt_name,
                description=prompt_desc,
                arguments=arguments,
                handler=func
            )
            
            return func
        return decorator
    
    def _handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a JSON-RPC request."""
        method = request.get("method", "")
        params = request.get("params", {})
        request_id = request.get("id")
        
        try:
            result = self._dispatch(method, params)
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32000,
                    "message": str(e)
                }
            }
    
    def _dispatch(self, method: str, params: Dict[str, Any]) -> Any:
        """Dispatch a method call."""
        
        if method == "initialize":
            return {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {"listChanged": False} if self.tools else None,
                    "resources": {"listChanged": False} if self.resources else None,
                    "prompts": {"listChanged": False} if self.prompts else None,
                },
                "serverInfo": {
                    "name": self.name,
                    "version": self.version
                }
            }
        
        elif method == "notifications/initialized":
            return {}
        
        elif method == "tools/list":
            return {
                "tools": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": tool.parameters
                    }
                    for tool in self.tools.values()
                ]
            }
        
        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            
            if tool_name not in self.tools:
                raise ValueError(f"Unknown tool: {tool_name}")
            
            tool = self.tools[tool_name]
            result = tool.function(**arguments)
            
            # Handle async functions
            if asyncio.iscoroutine(result):
                result = asyncio.get_event_loop().run_until_complete(result)
            
            # Format result
            if isinstance(result, str):
                content = [{"type": "text", "text": result}]
            elif isinstance(result, dict):
                content = [{"type": "text", "text": json.dumps(result)}]
            else:
                content = [{"type": "text", "text": str(result)}]
            
            return {"content": content}
        
        elif method == "resources/list":
            return {
                "resources": [
                    {
                        "uri": res.uri,
                        "name": res.name,
                        "description": res.description,
                        "mimeType": res.mime_type
                    }
                    for res in self.resources.values()
                ]
            }
        
        elif method == "resources/read":
            uri = params.get("uri")
            
            if uri not in self.resources:
                raise ValueError(f"Unknown resource: {uri}")
            
            resource = self.resources[uri]
            content = resource.handler()
            
            if asyncio.iscoroutine(content):
                content = asyncio.get_event_loop().run_until_complete(content)
            
            return {
                "contents": [{
                    "uri": uri,
                    "mimeType": resource.mime_type,
                    "text": content if isinstance(content, str) else json.dumps(content)
                }]
            }
        
        elif method == "prompts/list":
            return {
                "prompts": [
                    {
                        "name": prompt.name,
                        "description": prompt.description,
                        "arguments": prompt.arguments
                    }
                    for prompt in self.prompts.values()
                ]
            }
        
        elif method == "prompts/get":
            prompt_name = params.get("name")
            arguments = params.get("arguments", {})
            
            if prompt_name not in self.prompts:
                raise ValueError(f"Unknown prompt: {prompt_name}")
            
            prompt = self.prompts[prompt_name]
            content = prompt.handler(**arguments)
            
            if asyncio.iscoroutine(content):
                content = asyncio.get_event_loop().run_until_complete(content)
            
            # Format as messages
            if isinstance(content, str):
                messages = [{"role": "user", "content": {"type": "text", "text": content}}]
            elif isinstance(content, list):
                messages = content
            else:
                messages = [{"role": "user", "content": {"type": "text", "text": str(content)}}]
            
            return {"messages": messages}
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def run(self, transport: str = "stdio"):
        """
        Run the MCP server.
        
        Args:
            transport: "stdio" (default) or "http"
        """
        if transport == "stdio":
            self._run_stdio()
        elif transport == "http":
            self._run_http()
        else:
            raise ValueError(f"Unknown transport: {transport}")
    
    def _run_stdio(self):
        """Run server with stdio transport."""
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                
                request = json.loads(line)
                response = self._handle_request(request)
                
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
                
            except json.JSONDecodeError:
                continue
            except EOFError:
                break
            except KeyboardInterrupt:
                break
    
    def _run_http(self, host: str = "0.0.0.0", port: int = 8000):
        """Run server with HTTP transport using stdlib http.server."""
        from http.server import HTTPServer, BaseHTTPRequestHandler
        
        server_instance = self
        
        class MCPHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length)
                
                try:
                    request = json.loads(body)
                    response = server_instance._handle_request(request)
                    
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(response).encode())
                    
                except Exception as e:
                    self.send_response(500)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode())
            
            def log_message(self, format, *args):
                pass  # Suppress logging
        
        print(f"Starting MCP server '{self.name}' on http://{host}:{port}/mcp")
        httpd = HTTPServer((host, port), MCPHandler)
        httpd.serve_forever()


# =============================================================================
# EXAMPLE
# =============================================================================

def create_example_server():
    """Create an example MCP server."""
    server = MCPServer("Example")
    
    @server.tool()
    def add(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b
    
    @server.tool()
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers together."""
        return a * b
    
    @server.tool()
    def greet(name: str) -> str:
        """Greet someone by name."""
        return f"Hello, {name}!"
    
    @server.resource("file:///info.json", name="Server Info")
    def get_info():
        return {"name": server.name, "version": server.version}
    
    @server.prompt()
    def summarize(text: str) -> str:
        """Generate a summary prompt."""
        return f"Please summarize the following text in 2-3 sentences:\n\n{text}"
    
    return server


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="StoneChain MCP Server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--example", action="store_true", help="Run example server")
    args = parser.parse_args()
    
    if args.example:
        server = create_example_server()
        print(f"Running example server '{server.name}'")
        print(f"Tools: {list(server.tools.keys())}")
        server.run(transport=args.transport)
    else:
        print("StoneChain MCP Server Builder")
        print("=" * 40)
        print(f"Version: {__version__}")
        print("\nUsage:")
        print("  from stonechain_mcp_server import MCPServer")
        print("  ")
        print("  server = MCPServer('MyServer')")
        print("  ")
        print("  @server.tool()")
        print("  def add(a: int, b: int) -> int:")
        print("      '''Add two numbers'''")
        print("      return a + b")
        print("  ")
        print("  server.run()")
