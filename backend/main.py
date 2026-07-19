import os
import json
import shutil
import base64
import traceback
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from agent.rag import RagStore
from agent.mcp_client import McpClientManager
from agent.runner import run_agent

app = FastAPI(title="Interactive Chat Agent API")

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict to frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global stores
rag_store = RagStore()
mcp_manager = McpClientManager()

# Lifecycle events
@app.on_event("startup")
async def startup_event():
    print("Starting backend and connecting to configured MCP servers...")
    await mcp_manager.start_all()

@app.on_event("shutdown")
async def shutdown_event():
    print("Shutting down backend and disconnecting MCP servers...")
    await mcp_manager.stop_all()


# --- MCP Server Routes ---

class McpServerModel(BaseModel):
    name: str
    command: str
    args: List[str] = []
    env: Dict[str, str] = {}

@app.get("/api/mcp/servers")
async def list_mcp_servers():
    """List configured and connected MCP servers."""
    results = []
    for name, config in mcp_manager.configs.items():
        is_connected = name in mcp_manager.servers and mcp_manager.servers[name].process is not None
        tool_count = len(mcp_manager.servers[name].tools) if is_connected else 0
        results.append({
            "name": name,
            "command": config["command"],
            "args": config.get("args", []),
            "env": config.get("env", {}),
            "connected": is_connected,
            "tools_count": tool_count
        })
    return results

@app.post("/api/mcp/servers")
async def add_mcp_server(server: McpServerModel):
    """Add and connect a new MCP server."""
    success = await mcp_manager.add_server(
        name=server.name,
        command=server.command,
        args=server.args,
        env=server.env
    )
    if not success:
        # Configuration is saved, but connection failed
        return {"status": "configured_but_failed_connection", "message": f"Server {server.name} registered, but failed to connect/initialize."}
    
    return {"status": "success", "message": f"Server {server.name} connected successfully."}

@app.delete("/api/mcp/servers/{name}")
async def remove_mcp_server(name: str):
    """Remove and stop an MCP server."""
    await mcp_manager.remove_server(name)
    return {"status": "success", "message": f"Server {name} removed."}

@app.get("/api/mcp/tools")
async def list_mcp_tools():
    """List all available tools from all connected MCP servers."""
    return mcp_manager.get_all_tools()


class ImageGenerationRequest(BaseModel):
    prompt: str
    model: str = ""
    n: int = 1
    size: str = "1024x1024"
    api_type: str = "lmstudio"
    api_url: str = "http://host.docker.internal:11434"
    api_key: str = ""


def _parse_image_response(response_data: Any) -> List[Dict[str, str]]:
    images = []
    if isinstance(response_data, dict):
        candidates = response_data.get("data") or response_data.get("output") or []
        if isinstance(candidates, dict):
            candidates = [candidates]
        if isinstance(candidates, list):
            for item in candidates:
                if not isinstance(item, dict):
                    continue
                if item.get("b64_json"):
                    images.append({
                        "mime_type": "image/png",
                        "base64": item["b64_json"]
                    })
                elif isinstance(item.get("image_url"), str):
                    images.append({
                        "mime_type": "image/png",
                        "url": item["image_url"]
                    })
                elif isinstance(item.get("url"), str):
                    images.append({
                        "mime_type": "image/png",
                        "url": item["url"]
                    })
    return images


@app.post("/api/images/generate")
async def generate_image(request: ImageGenerationRequest):
    """Generate an image using the configured local model endpoint."""
    headers = {}
    if request.api_key:
        headers["Authorization"] = f"Bearer {request.api_key}"

    payload = {
        "prompt": request.prompt,
        "n": request.n,
        "size": request.size
    }
    if request.model:
        payload["model"] = request.model

    endpoint = f"{request.api_url.rstrip('/')}/v1/images/generate"

    try:
        timeout = httpx.Timeout(connect=10.0, read=None, write=60.0, pool=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(endpoint, json=payload, headers=headers)
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=f"Image generation error: {response.text}")
            data = response.json()
            images = _parse_image_response(data)
            if not images:
                raise HTTPException(status_code=500, detail="No image data returned from model.")
            return {
                "status": "success",
                "images": images,
                "raw": data
            }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate image: {str(e)}")


# --- RAG / Document Routes ---

@app.get("/api/rag/status")
async def get_rag_status():
    """Get indexing status of documents."""
    doc_list = list(rag_store.documents.keys())
    chunks_count = len(rag_store.chunks)
    unembedded_chunks = len([c for c in rag_store.chunks if c.embedding is None])
    
    return {
        "documents": doc_list,
        "total_chunks": chunks_count,
        "indexed_chunks": chunks_count - unembedded_chunks,
        "unembedded_chunks": unembedded_chunks
    }

@app.post("/api/rag/upload")
async def upload_document(
    file: UploadFile = File(...),
    embedding_type: str = Form("ollama"),
    embedding_url: str = Form("http://host.docker.internal:11434"),
    embedding_model: str = Form(""),
    api_key: str = Form("")
):
    """Upload and index a text/markdown/PDF document."""
    try:
        filename = file.filename
        content_type = file.content_type
        
        text = ""
        # 1. Parse content based on file type
        if filename.endswith(".pdf"):
            # Use pypdf to parse PDF content
            from pypdf import PdfReader
            import io
            file_bytes = await file.read()
            pdf_file = io.BytesIO(file_bytes)
            reader = PdfReader(pdf_file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        else:
            # Assume text/markdown/code file
            file_bytes = await file.read()
            text = file_bytes.decode("utf-8", errors="ignore")
            
        if not text.strip():
            raise HTTPException(status_code=400, detail="The uploaded file contains no extractable text.")
            
        # 2. Add document to RAG store
        rag_store.add_document(doc_name=filename, text=text)
        
        # 3. Asynchronously compute embeddings if model details are provided
        if embedding_model:
            # Trigger embedding generation in background task
            # (We run it in asyncio to not block the FastAPI loop)
            asyncio.create_task(rag_store.compute_embeddings(
                api_type=embedding_type,
                api_url=embedding_url,
                model_name=embedding_model,
                api_key=api_key
            ))
            
        return {
            "status": "success",
            "message": f"Document '{filename}' indexed successfully. Chunking complete. Embeddings generation triggered in the background.",
            "doc_name": filename,
            "chunks_created": len([c for c in rag_store.chunks if c.doc_name == filename])
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to process upload: {str(e)}")

@app.post("/api/rag/clear")
async def clear_rag():
    """Clear all indexed documents."""
    rag_store.clear()
    return {"status": "success", "message": "All documents cleared from library."}


@app.get("/api/models")
async def list_models(
    api_type: str = "lmstudio",
    api_url: str = "http://host.docker.internal:51234",
    x_llm_api_key: str = Header(default="", alias="X-LLM-Api-Key")
):
    """Retrieve loaded model names from the configured LLM endpoint.

    The API key is read from a request header rather than a query
    parameter — query strings end up in access logs (and browser/proxy
    history) in plaintext, headers don't.
    """
    headers = {}
    if x_llm_api_key:
        headers["Authorization"] = f"Bearer {x_llm_api_key}"

    candidates = ["/v1/models", "/models", "/api/models"]
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=15.0, write=10.0, pool=5.0)) as client:
        for path in candidates:
            try:
                url = f"{api_url.rstrip('/')}{path}"
                response = await client.get(url, headers=headers)
                if response.status_code != 200:
                    continue
                data = response.json()
                models = []
                if isinstance(data, dict):
                    if "models" in data and isinstance(data["models"], list):
                        models = data["models"]
                    elif "data" in data and isinstance(data["data"], list):
                        models = data["data"]
                    elif "model" in data and isinstance(data["model"], (str, dict)):
                        models = [data["model"]]
                elif isinstance(data, list):
                    models = data

                parsed = []
                for item in models:
                    if isinstance(item, str):
                        parsed.append(item)
                    elif isinstance(item, dict):
                        parsed.append(item.get("id") or item.get("name") or json.dumps(item))
                    else:
                        parsed.append(str(item))

                return {"success": True, "models": parsed}
            except Exception:
                continue

    raise HTTPException(status_code=502, detail="Unable to fetch loaded models from the configured endpoint.")


# --- WebSocket Streaming Agent Router ---

@app.websocket("/api/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket client connected.")
    
    try:
        while True:
            # Receive data from frontend
            data = await websocket.receive_text()
            req = json.loads(data)
            
            query = req.get("query", "")
            images = req.get("images", []) # List of {"mime_type": ..., "base64": ...}
            history = req.get("history", [])
            config = req.get("config", {})
            
            print(f"Executing agent query: {query[:50]}...")
            
            # Execute runner
            # We iterate through the runner generator and send messages to WS in real-time
            try:
                async for event in run_agent(
                    query=query,
                    images=images,
                    history=history,
                    config=config,
                    rag_store=rag_store,
                    mcp_manager=mcp_manager
                ):
                    # Send event to websocket client
                    await websocket.send_json(event)
            except WebSocketDisconnect:
                # Client disconnected mid-stream (tab closed/reloaded) — nothing to notify, let
                # the outer handler log it once instead of also trying (and failing) to send here.
                raise
            except Exception as e:
                traceback.print_exc()
                try:
                    await websocket.send_json({
                        "event": "error",
                        "content": f"Internal agent execution error: {str(e)}"
                    })
                except Exception:
                    # Socket died right as we were reporting the error — client is already gone.
                    pass
                
    except WebSocketDisconnect:
        print("WebSocket client disconnected.")
    except Exception as e:
        print(f"WebSocket connection error: {e}")
        try:
            await websocket.close()
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
