#!/usr/bin/env python3
"""
Mock MCP Server

A simple HTTP server that mocks the MCP (Model Composition Platform) API.
The server provides endpoints for tool discovery and execution.
"""
import threading
import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, List, Optional, Callable
import os
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockMCPRequestHandler(BaseHTTPRequestHandler):
    """Handler for mock MCP server requests."""

    def _send_response(self, status_code: int, response_data: Dict[str, Any]):
        """Send a JSON response."""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response_data).encode('utf-8'))

    def _parse_json_body(self) -> Dict[str, Any]:
        """Parse the request body as JSON."""
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8')
        return json.loads(body) if body else {}

    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/tools':
            tools = self.server.get_tools()
            self._send_response(200, {'tools': tools})
        else:
            self._send_response(404, {'error': 'Not found'})

    def do_POST(self):
        """Handle POST requests."""
        if self.path == '/execute':
            try:
                request_data = self._parse_json_body()
                tool_name = request_data.get('name')
                parameters = request_data.get('parameters', {})
                
                if not tool_name:
                    self._send_response(400, {'error': 'Missing tool name'})
                    return
                
                result = self.server.execute_tool(tool_name, parameters)
                self._send_response(200, result)
            except Exception as e:
                logger.error(f"Error executing tool: {e}")
                self._send_response(500, {'error': str(e)})
        else:
            self._send_response(404, {'error': 'Not found'})

    def log_message(self, format, *args):
        """Customize logging for the server."""
        logger.debug(f"MockMCPServer: {self.address_string()} - {format % args}")


class MockMCPServer(HTTPServer):
    """Mock MCP server that handles tool discovery and execution."""

    def __init__(self, host: str = 'localhost', port: int = 8000):
        """Initialize the mock MCP server.
        
        Args:
            host: The hostname to bind to
            port: The port to bind to
        """
        super().__init__((host, port), MockMCPRequestHandler)
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.tools = self._create_mock_tools()
        self.server_thread = None
        self.running = False
        
    def _create_mock_tools(self) -> List[Dict[str, Any]]:
        """Create mock tools for the server."""
        return [
            {
                "name": "calculator_compute",
                "description": "Perform arithmetic calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The arithmetic expression to evaluate"
                        }
                    },
                    "required": ["expression"]
                }
            },
            {
                "name": "weather_get_forecast",
                "description": "Get weather forecast for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The location to get weather for"
                        }
                    },
                    "required": ["location"]
                }
            },
            {
                "name": "stocks_get_price",
                "description": "Get current stock price for a ticker symbol",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "The stock ticker symbol"
                        }
                    },
                    "required": ["symbol"]
                }
            },
            {
                "name": "web_search",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        }
                    },
                    "required": ["query"]
                }
            }
        ]
        
    def get_tools(self) -> List[Dict[str, Any]]:
        """Get all available tools."""
        return self.tools
    
    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool with the given parameters.
        
        Args:
            tool_name: The name of the tool to execute
            parameters: The parameters to pass to the tool
            
        Returns:
            The result of the tool execution
            
        Raises:
            ValueError: If the tool is not found
        """
        if tool_name == "calculator_compute":
            expression = parameters.get("expression", "")
            try:
                # Very limited safety - for a real implementation, use a safer evaluation method
                result = eval(expression, {"__builtins__": {}})
                return {"result": {"result": result}}
            except Exception as e:
                return {"error": f"Failed to evaluate expression: {str(e)}"}
                
        elif tool_name == "weather_get_forecast":
            location = parameters.get("location", "")
            # Mock weather data
            return {
                "forecast": {
                    "location": location,
                    "current": {
                        "temperature": 72,
                        "condition": "Partly Cloudy",
                        "humidity": 65,
                        "wind_speed": 5
                    },
                    "daily": [
                        {"day": "Today", "high": 75, "low": 60, "condition": "Partly Cloudy"},
                        {"day": "Tomorrow", "high": 78, "low": 62, "condition": "Sunny"},
                        {"day": "Saturday", "high": 80, "low": 65, "condition": "Sunny"}
                    ]
                }
            }
            
        elif tool_name == "stocks_get_price":
            symbol = parameters.get("symbol", "")
            # Mock stock data
            prices = {
                "AAPL": 185.92,
                "MSFT": 420.45,
                "GOOG": 176.75,
                "AMZN": 185.35,
                "META": 500.23,
                "TSLA": 175.47
            }
            price = prices.get(symbol.upper(), 100.00)
            
            return {
                "stock": {
                    "symbol": symbol.upper(),
                    "price": price,
                    "currency": "USD",
                    "change": 1.25,
                    "change_percent": 0.7
                }
            }
            
        elif tool_name == "web_search":
            query = parameters.get("query", "")
            # Mock search results
            return {
                "search_results": {
                    "query": query,
                    "results": [
                        {
                            "title": f"Result 1 for {query}",
                            "url": f"https://example.com/result1?q={query}",
                            "snippet": f"This is a snippet for result 1 about {query}. It contains relevant information."
                        },
                        {
                            "title": f"Result 2 for {query}",
                            "url": f"https://example.org/result2?q={query}",
                            "snippet": f"Another snippet for result 2 about {query}. More details about the topic."
                        },
                        {
                            "title": f"Result 3 for {query}",
                            "url": f"https://example.net/result3?q={query}",
                            "snippet": f"Third snippet for result 3 about {query}. Additional context and information."
                        }
                    ]
                }
            }
        
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    
    def start(self):
        """Start the server in a separate thread."""
        if self.running:
            logger.warning("Server is already running")
            return
            
        def run_server():
            logger.info(f"Starting mock MCP server at {self.base_url}")
            self.running = True
            self.serve_forever()
            
        self.server_thread = threading.Thread(target=run_server)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        # Give the server a moment to start
        time.sleep(0.1)
    
    def stop(self):
        """Stop the server."""
        if not self.running:
            logger.warning("Server is not running")
            return
            
        logger.info("Stopping mock MCP server")
        self.shutdown()
        self.server_close()
        self.running = False
        
        if self.server_thread:
            self.server_thread.join(1.0)


# Example standalone usage
if __name__ == "__main__":
    # Start the server
    server = MockMCPServer()
    print(f"Starting mock MCP server at {server.base_url}")
    print("Press Ctrl+C to stop")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping server...")
    finally:
        server.server_close()
        print("Server stopped") 