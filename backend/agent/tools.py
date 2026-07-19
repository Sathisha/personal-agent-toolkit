import os
import subprocess
import sys
import httpx
import traceback
import json
from typing import Dict, Any, List

WORKSPACE_DIR = os.environ.get("WORKSPACE_DIR", "/workspace_files")
SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://searxng:8080")

# Marker prefix so callers (runner.py) can detect a connectivity failure
# programmatically, without having to guess from free-form error text.
NO_INTERNET_PREFIX = "NO_INTERNET:"

# Ensure workspace dir exists
os.makedirs(WORKSPACE_DIR, exist_ok=True)

def safe_path(path: str) -> str:
    # Resolve absolute path and ensure it's inside the workspace directory
    abs_workspace = os.path.abspath(WORKSPACE_DIR)
    # Handle relative or absolute paths input by the LLM
    if os.path.isabs(path):
        # If the LLM gives an absolute path like /workspace_files/test.txt or C:/test.txt, try to keep it local to workspace
        rel = os.path.relpath(path, abs_workspace)
        if rel.startswith("..") or path.startswith("/.."):
            raise ValueError("Access denied: path is outside the workspace directory.")
        return os.path.abspath(path)
    else:
        target_path = os.path.abspath(os.path.join(abs_workspace, path))
        if not target_path.startswith(abs_workspace):
            raise ValueError("Access denied: path is outside the workspace directory.")
        return target_path

# Tool implementations

def read_file(path: str) -> str:
    """Read the contents of a file in the workspace."""
    try:
        real_path = safe_path(path)
        if not os.path.exists(real_path):
            return f"Error: File not found at {path}"
        with open(real_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        return content
    except Exception as e:
        return f"Error reading file: {str(e)}"

def write_file(path: str, content: str) -> str:
    """Write content to a file in the workspace."""
    try:
        real_path = safe_path(path)
        os.makedirs(os.path.dirname(real_path), exist_ok=True)
        with open(real_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Success: File written to {path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"

def run_command(command: str) -> str:
    """Run a shell command inside the workspace directory and return output."""
    try:
        # Run process inside workspace directory
        result = subprocess.run(
            command,
            shell=True,
            cwd=WORKSPACE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30
        )
        output = ""
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}\n"
        if result.stderr:
            output += f"STDERR:\n{result.stderr}\n"
        if not output:
            output = "Command executed successfully with no output."
        return output
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds."
    except Exception as e:
        return f"Error executing command: {str(e)}"

def python_interpreter(code: str) -> str:
    """Execute python code dynamically and return the printed output."""
    try:
        # Run code in a subprocess to avoid crashing the backend server and ensure clean environment
        # We can write the code to a temp file and execute it
        temp_file = os.path.join(WORKSPACE_DIR, "_temp_exec.py")
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(code)
        
        result = subprocess.run(
            [sys.executable, temp_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15
        )
        
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += f"Error output:\n{result.stderr}"
        if not output:
            output = "Code executed successfully with no output."
        return output
    except subprocess.TimeoutExpired:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return "Error: Execution timed out after 15 seconds."
    except Exception as e:
        return f"Error running python code: {str(e)}"

async def search_web(query: str) -> str:
    """Search the web via a self-hosted SearXNG instance (no API keys required)."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(
                f"{SEARXNG_URL.rstrip('/')}/search",
                params={"q": query, "format": "json"}
            )
            if response.status_code != 200:
                return f"Error searching the web: SearXNG returned HTTP {response.status_code}. Is the searxng container running?"

            data = response.json()
            results = data.get("results", [])[:5]
            if not results:
                return "No search results found. Try a broader or differently-worded query."

            formatted = []
            for item in results:
                title = item.get("title", "No Title")
                url = item.get("url", "")
                snippet = item.get("content", "")
                formatted.append(f"Title: {title}\nURL: {url}\nSnippet: {snippet}\n")

            return "\n---\n".join(formatted)
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.TransportError) as e:
        # Distinguish "can't reach the network" from other failures so the caller can back off
        # gracefully instead of endlessly retrying a search that can never succeed this turn.
        return (
            f"{NO_INTERNET_PREFIX} Could not reach the search backend at {SEARXNG_URL}. "
            f"This usually means there is no internet connection right now, or the searxng "
            f"container isn't running. Details: {str(e)}"
        )
    except Exception as e:
        return f"Error executing web search: {str(e)}"

# Definitions of tools schema for the LLM

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a text file inside the workspace directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The relative path to the file inside the workspace."
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file in the workspace with new content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The relative path to the file to create/overwrite."
                    },
                    "content": {
                        "type": "string",
                        "description": "The full text content to write to the file."
                    }
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a terminal shell command (bash/cmd) inside the workspace directory and return the output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute."
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "python_interpreter",
            "description": "Execute Python code in an isolated script inside the workspace. Use this for computations, data processing, math, or scripting.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to execute. Standard print() calls will be returned as output."
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Search the web for up-to-date information, news, or general knowledge. "
                "Results are often incomplete from a single query — read what comes back, "
                "and if it doesn't fully answer the question, call this tool again with a "
                "more specific or differently-worded query (e.g. narrow by date, operator "
                "name, or a term you learned from the previous results). Repeat this up to "
                "several times until you've gathered enough reliable information, then "
                "synthesize a final answer. If a result starts with 'NO_INTERNET:', stop "
                "searching immediately — there is no connectivity right now — and tell the "
                "user clearly rather than retrying."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_documents",
            "description": "Query the uploaded RAG documents database for relevant context regarding the files/documents uploaded by the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The semantic search query for the document library."
                    }
                },
                "required": ["query"]
            }
        }
    }
]

async def execute_tool(name: str, arguments: Dict[str, Any], rag_search_func=None) -> str:
    """Execute tool by name and arguments."""
    try:
        if name == "read_file":
            return read_file(arguments.get("path", ""))
        elif name == "write_file":
            return write_file(arguments.get("path", ""), arguments.get("content", ""))
        elif name == "run_command":
            return run_command(arguments.get("command", ""))
        elif name == "python_interpreter":
            return python_interpreter(arguments.get("code", ""))
        elif name == "search_web":
            return await search_web(arguments.get("query", ""))
        elif name == "query_documents":
            if rag_search_func:
                return await rag_search_func(arguments.get("query", ""))
            else:
                return "RAG is not initialized or no documents are currently index."
        else:
            return f"Error: Tool '{name}' not found."
    except Exception as e:
        return f"Error executing tool '{name}': {str(e)}\n{traceback.format_exc()}"
