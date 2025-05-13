"""
Mock MCP Server

This module provides a simple mock MCP server for testing purposes.
It simulates an MCP endpoint with tools, responses, and proper API format.
"""
import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define mock MCP tools
MOCK_MCP_TOOLS = {
    "weather_get_forecast": {
        "name": "weather_get_forecast",
        "description": "Get weather forecast for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The location to get weather for (city, zip code, etc.)"
                },
                "days": {
                    "type": "integer", 
                    "description": "Number of days to forecast"
                }
            },
            "required": ["location"]
        }
    },
    "calculator_compute": {
        "name": "calculator_compute",
        "description": "Compute a mathematical expression",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to compute"
                }
            },
            "required": ["expression"]
        }
    },
    "stocks_get_price": {
        "name": "stocks_get_price",
        "description": "Get the current stock price",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "The stock symbol (e.g. AAPL, MSFT)"
                }
            },
            "required": ["symbol"]
        }
    },
    "web_search": {
        "name": "web_search",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return"
                }
            },
            "required": ["query"]
        }
    }
}

# Define mock responses for each tool
MOCK_RESPONSES = {
    "weather_get_forecast": lambda params: {
        "forecast": f"Weather forecast for {params.get('location', 'Unknown')}: Sunny, 75°F",
        "daily": [
            {"day": 1, "condition": "Sunny", "high": 75, "low": 60},
            {"day": 2, "condition": "Partly Cloudy", "high": 72, "low": 58},
            {"day": 3, "condition": "Rain", "high": 65, "low": 55}
        ]
    },
    "calculator_compute": lambda params: {
        "expression": params.get("expression", ""),
        "result": eval(params.get("expression", "0"), {"__builtins__": {}}, {"abs": abs, "max": max, "min": min, "sum": sum})
    },
    "stocks_get_price": lambda params: {
        "symbol": params.get("symbol", "Unknown"),
        "price": 150.25 if params.get("symbol") == "AAPL" else 250.75,
        "change": "+1.25%",
        "volume": "10.5M"
    },
    "web_search": lambda params: {
        "query": params.get("query", ""),
        "results": [
            {"title": f"Result 1 for {params.get('query', '')}", "url": "https://example.com/1", "snippet": "This is the first result."},
            {"title": f"Result 2 for {params.get('query', '')}", "url": "https://example.com/2", "snippet": "This is the second result."},
            {"title": f"Result 3 for {params.get('query', '')}", "url": "https://example.com/3", "snippet": "This is the third result."}
        ]
    }
}


class MockMCPRequestHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler for the Mock MCP Server."""
    
    def do_GET(self):
        """Handle GET requests for tool discovery."""
        if self.path == "/tools":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            
            response = {
                "tools": list(MOCK_MCP_TOOLS.values())
            }
            
            self.wfile.write(json.dumps(response).encode("utf-8"))
            logger.info("Served tool list")
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not found")
    
    def do_POST(self):
        """Handle POST requests for tool execution."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        request_data = json.loads(body)
        
        if self.path == "/execute":
            tool_name = request_data.get("name")
            parameters = request_data.get("parameters", {})
            
            if tool_name in MOCK_RESPONSES:
                try:
                    # Simulate some processing time
                    time.sleep(0.2)
                    
                    result = MOCK_RESPONSES[tool_name](parameters)
                    
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    
                    response = {
                        "result": result
                    }
                    
                    self.wfile.write(json.dumps(response).encode("utf-8"))
                    logger.info(f"Executed tool: {tool_name} with params: {parameters}")
                except Exception as e:
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    
                    response = {
                        "error": f"Error executing tool: {str(e)}"
                    }
                    
                    self.wfile.write(json.dumps(response).encode("utf-8"))
                    logger.error(f"Error executing tool: {e}")
            else:
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                
                response = {
                    "error": f"Tool not found: {tool_name}"
                }
                
                self.wfile.write(json.dumps(response).encode("utf-8"))
                logger.warning(f"Tool not found: {tool_name}")
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not found")


class MockMCPServer:
    """A mock MCP server for testing purposes."""
    
    def __init__(self, host="localhost", port=8000):
        """Initialize the server with the given host and port."""
        self.host = host
        self.port = port
        self.server = None
        self.server_thread = None
        self.is_running = False
    
    def start(self):
        """Start the mock MCP server."""
        if self.is_running:
            logger.warning("Server is already running")
            return
        
        self.server = HTTPServer((self.host, self.port), MockMCPRequestHandler)
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()
        self.is_running = True
        
        logger.info(f"Mock MCP server started at http://{self.host}:{self.port}")
    
    def stop(self):
        """Stop the mock MCP server."""
        if not self.is_running:
            logger.warning("Server is not running")
            return
        
        self.server.shutdown()
        self.server.server_close()
        self.server_thread.join()
        self.is_running = False
        
        logger.info("Mock MCP server stopped")
    
    @property
    def base_url(self):
        """Get the base URL of the server."""
        return f"http://{self.host}:{self.port}"
    
    @property
    def tools_url(self):
        """Get the URL for tool discovery."""
        return f"{self.base_url}/tools"
    
    @property
    def execute_url(self):
        """Get the URL for tool execution."""
        return f"{self.base_url}/execute"


if __name__ == "__main__":
    # Start the mock MCP server for testing
    server = MockMCPServer(port=8000)
    server.start()
    
    print(f"Mock MCP server running at {server.base_url}")
    print("Available tools:")
    for tool_name, tool_def in MOCK_MCP_TOOLS.items():
        print(f"  - {tool_name}: {tool_def['description']}")
    
    print("\nPress Ctrl+C to stop the server...")
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping server...")
        server.stop()
        print("Server stopped") 