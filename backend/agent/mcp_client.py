import os
import json
import asyncio
import traceback
from typing import Dict, Any, List, Optional

CONFIG_FILE = os.path.join(os.environ.get("WORKSPACE_DIR", "/workspace_files"), "mcp_servers.json")

class McpServerConnection:
    def __init__(self, name: str, command: str, args: List[str] = None, env: Dict[str, str] = None):
        self.name = name
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.process: Optional[asyncio.subprocess.Process] = None
        self.msg_id = 0
        self.pending_requests: Dict[int, asyncio.Future] = {}
        self.tools: List[Dict[str, Any]] = []
        self.read_task: Optional[asyncio.Task] = None

    async def connect(self):
        try:
            # Prepare environment variables
            full_env = os.environ.copy()
            full_env.update(self.env)
            
            # Combine command and args
            cmd = [self.command] + self.args
            
            print(f"Connecting to MCP server '{self.name}' via command: {' '.join(cmd)}")
            
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=full_env
            )
            
            # Start background reader task
            self.read_task = asyncio.create_task(self._read_loop())
            
            # Perform initialize handshake
            success = await self._initialize()
            if success:
                print(f"MCP server '{self.name}' initialized successfully.")
                # Load tools
                await self.refresh_tools()
            else:
                print(f"MCP server '{self.name}' failed initialization.")
                await self.disconnect()
        except Exception as e:
            print(f"Error connecting to MCP server '{self.name}': {e}\n{traceback.format_exc()}")
            await self.disconnect()

    async def disconnect(self):
        if self.read_task:
            self.read_task.cancel()
            self.read_task = None
            
        if self.process:
            try:
                self.process.terminate()
                await self.process.wait()
            except Exception:
                pass
            self.process = None
            
        self.tools = []
        # Cancel any pending requests
        for fut in self.pending_requests.values():
            if not fut.done():
                fut.set_exception(Exception("Disconnected from server"))
        self.pending_requests.clear()
        print(f"Disconnected from MCP server '{self.name}'.")

    async def _send_request(self, method: str, params: Dict[str, Any]) -> Any:
        if not self.process or not self.process.stdin:
            raise Exception("Server not connected")
            
        self.msg_id += 1
        req_id = self.msg_id
        
        request = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params
        }
        
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self.pending_requests[req_id] = fut
        
        # Write line to stdin
        req_str = json.dumps(request) + "\n"
        self.process.stdin.write(req_str.encode('utf-8'))
        await self.process.stdin.drain()
        
        # Wait for response (with timeout)
        try:
            return await asyncio.wait_for(fut, timeout=10.0)
        except asyncio.TimeoutExpired:
            self.pending_requests.pop(req_id, None)
            raise Exception("Request timed out")

    async def _send_notification(self, method: str, params: Dict[str, Any] = None):
        if not self.process or not self.process.stdin:
            return
            
        notification = {
            "jsonrpc": "2.0",
            "method": method
        }
        if params is not None:
            notification["params"] = params
            
        notif_str = json.dumps(notification) + "\n"
        self.process.stdin.write(notif_str.encode('utf-8'))
        await self.process.stdin.drain()

    async def _initialize(self) -> bool:
        try:
            params = {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "InteractiveChatAgentClient",
                    "version": "1.0.0"
                }
            }
            res = await self._send_request("initialize", params)
            
            # Send initialized notification
            await self._send_notification("notifications/initialized")
            return True
        except Exception as e:
            print(f"Initialization handshake failed for '{self.name}': {e}")
            return False

    async def refresh_tools(self):
        try:
            res = await self._send_request("tools/list", {})
            self.tools = res.get("tools", [])
            print(f"Loaded {len(self.tools)} tools from MCP server '{self.name}'.")
        except Exception as e:
            print(f"Failed to list tools for '{self.name}': {e}")
            self.tools = []

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        try:
            params = {
                "name": name,
                "arguments": arguments
            }
            res = await self._send_request("tools/call", params)
            return res
        except Exception as e:
            return {
                "isError": True,
                "content": [{"type": "text", "text": f"Error calling MCP tool '{name}': {str(e)}"}]
            }

    async def _read_loop(self):
        try:
            while self.process and self.process.stdout:
                line = await self.process.stdout.readline()
                if not line:
                    break
                
                try:
                    msg = json.loads(line.decode('utf-8').strip())
                    msg_id = msg.get("id")
                    
                    if msg_id is not None:
                        fut = self.pending_requests.pop(msg_id, None)
                        if fut and not fut.done():
                            if "error" in msg:
                                fut.set_exception(Exception(msg["error"].get("message", "Unknown error")))
                            else:
                                fut.set_result(msg.get("result"))
                    else:
                        # This is a notification or request from server (not handled for now)
                        pass
                except Exception as e:
                    print(f"Error parsing line from '{self.name}': {e}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Error in read loop for '{self.name}': {e}")


class McpClientManager:
    def __init__(self):
        self.servers: Dict[str, McpServerConnection] = {}
        self.configs: Dict[str, Dict[str, Any]] = {}
        self.load_configs()

    def load_configs(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    self.configs = json.load(f)
            except Exception as e:
                print(f"Error loading MCP configs: {e}")
                self.configs = {}
        else:
            self.configs = {}

    def save_configs(self):
        try:
            os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.configs, f, indent=2)
        except Exception as e:
            print(f"Error saving MCP configs: {e}")

    async def add_server(self, name: str, command: str, args: List[str] = None, env: Dict[str, str] = None) -> bool:
        self.configs[name] = {
            "command": command,
            "args": args or [],
            "env": env or {}
        }
        self.save_configs()
        
        # Stop existing connection if any
        if name in self.servers:
            await self.servers[name].disconnect()
            
        # Create and connect new server
        server = McpServerConnection(name, command, args, env)
        await server.connect()
        self.servers[name] = server
        return server.process is not None

    async def remove_server(self, name: str):
        if name in self.servers:
            await self.servers[name].disconnect()
            del self.servers[name]
        if name in self.configs:
            del self.configs[name]
            self.save_configs()

    async def start_all(self):
        for name, config in self.configs.items():
            if name not in self.servers:
                server = McpServerConnection(
                    name=name,
                    command=config["command"],
                    args=config.get("args", []),
                    env=config.get("env", {})
                )
                await server.connect()
                self.servers[name] = server

    async def stop_all(self):
        for server in list(self.servers.values()):
            await server.disconnect()
        self.servers.clear()

    def get_all_tools(self) -> List[Dict[str, Any]]:
        all_tools = []
        for server_name, server in self.servers.items():
            for tool in server.tools:
                # We prefix tool names to avoid collision: e.g. "postgres_query"
                prefixed_tool = tool.copy()
                prefixed_tool["name"] = f"{server_name}__{tool['name']}"
                prefixed_tool["description"] = f"[MCP Server: {server_name}] {tool.get('description', '')}"
                all_tools.append(prefixed_tool)
        return all_tools

    async def call_mcp_tool(self, full_tool_name: str, arguments: Dict[str, Any]) -> str:
        if "__" not in full_tool_name:
            return f"Error: Invalid MCP tool name format '{full_tool_name}'."
            
        server_name, actual_tool_name = full_tool_name.split("__", 1)
        if server_name not in self.servers:
            return f"Error: MCP Server '{server_name}' is not connected."
            
        res = await self.servers[server_name].call_tool(actual_tool_name, arguments)
        
        # Parse standard MCP tool call response
        # It usually contains a "content" list with "type": "text" elements
        if "isError" in res and res["isError"]:
            err_msg = ""
            for item in res.get("content", []):
                if item.get("type") == "text":
                    err_msg += item.get("text", "")
            return f"MCP Tool Execution Error:\n{err_msg or str(res)}"
            
        output_parts = []
        for item in res.get("content", []):
            if item.get("type") == "text":
                output_parts.append(item.get("text", ""))
            elif item.get("type") == "image":
                # handle image (could be base64 data)
                output_parts.append(f"[Image Returned: {item.get('mimeType', 'unknown')}]")
        
        return "\n".join(output_parts)
