import httpx
import json
import asyncio
import traceback
from typing import List, Dict, Any, AsyncGenerator, Optional
from .tools import TOOL_DEFINITIONS, execute_tool, NO_INTERNET_PREFIX


def _parse_data_uri(uri: str) -> Optional[Dict[str, str]]:
    if not uri.startswith("data:"):
        return None
    parts = uri.split(",", 1)
    if len(parts) != 2:
        return None
    header, payload = parts
    mime = header.split(";")[0].replace("data:", "")
    return {"mime_type": mime or "image/png", "base64": payload}


# Smaller instruction-tuned models tend to fall back on a trained-in "I don't have
# real-time access" refusal even when a working search_web tool is available and the
# system prompt tells them to use it. These phrases catch that specific pattern so we
# can force one corrective retry instead of just accepting the guess as final.
#
# This also has to catch the quieter failure mode: instead of refusing outright, the
# model just answers confidently from stale training knowledge and hedges with a
# knowledge-cutoff disclaimer (e.g. "does the latest version of X support Y" answered
# from memory). That's just as much a case for an actual search as an explicit refusal.
_CAPABILITY_REFUSAL_PHRASES = [
    "i do not have access to real-time",
    "i don't have access to real-time",
    "i do not have access to live",
    "i don't have access to live",
    "i do not have access to current",
    "i don't have access to current",
    "cannot check the specific availability",
    "cannot check availability",
    "can't check availability",
    "i cannot browse the internet",
    "i don't have the ability to browse",
    "i do not have the ability to browse",
    "i do not have real-time",
    "i don't have real-time",
    "no real-time data",
    "no access to real-time",
    "i am unable to access the internet",
    "i'm unable to access the internet",
    "i do not have internet access",
    "i don't have internet access",
    "as of my last update",
    "as of my last training",
    "as of my knowledge cutoff",
    "as of my training cutoff",
    "based on my training data",
    "my training data only goes up",
    "i don't have the most recent",
    "i do not have the most recent",
    "i don't have up-to-date information",
    "i do not have up-to-date information",
    "may have changed since",
    "might have changed since",
    "i'm not certain of the latest",
    "i am not certain of the latest",
]


def _looks_like_capability_refusal(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in _CAPABILITY_REFUSAL_PHRASES)


async def run_agent(
    query: str,
    images: List[Dict[str, str]], # list of {"mime_type": ..., "base64": ...}
    history: List[Dict[str, Any]],
    config: Dict[str, Any],
    rag_store: Any,
    mcp_manager: Any
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Main agentic runner. Streams reasoning, tools, and final response chunks.
    Loops back to LLM if tools are called.
    """
    api_type = config.get("api_type", "ollama") # ollama, lmstudio, openai
    api_url = config.get("api_url", "http://host.docker.internal:11434")
    model_name = config.get("model_name", "")
    api_key = config.get("api_key", "")
    
    print(f"Agent starting with LLM config - Type: {api_type}, URL: {api_url}, Model: {model_name}")
    system_prompt = config.get("system_prompt", "You are Personal Agent, a helpful agentic coding and productivity assistant.")
    use_tools = config.get("use_tools", True)

    # RAG settings for query_documents
    embedding_type = config.get("embedding_type", "ollama")
    embedding_url = config.get("embedding_url", "http://host.docker.internal:11434")
    embedding_model = config.get("embedding_model", "")
    
    # 1. Compile tools
    tools = []
    if use_tools:
        # Standard built-in tools
        tools.extend(TOOL_DEFINITIONS)
        # MCP tools
        mcp_tools = mcp_manager.get_all_tools()
        tools.extend(mcp_tools)

    # Convert tools to schema (OpenAI format)
    openai_tools = None
    if tools:
        # Note: if a model doesn't support tools, it might error out, so we only pass if tools exist
        openai_tools = tools

    # 2. Build Message List
    # Setup initial messages list
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add history
    for msg in history:
        # Clean history messages to match OpenAI standard
        clean_msg = {"role": msg["role"]}
        if msg.get("content") is not None:
            clean_msg["content"] = msg["content"]
        if msg.get("tool_calls") is not None:
            clean_msg["tool_calls"] = msg["tool_calls"]
        if msg.get("tool_call_id") is not None:
            clean_msg["tool_call_id"] = msg["tool_call_id"]
        if msg.get("name") is not None:
            clean_msg["name"] = msg["name"]
        messages.append(clean_msg)

    # Build current user message (handling vision/images)
    user_content = []
    if query:
        user_content.append({"type": "text", "text": query})
    for img in images:
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{img['mime_type']};base64,{img['base64']}"
            }
        })
    
    # If no images, we can just use normal string content
    if len(user_content) == 1 and user_content[0]["type"] == "text":
        messages.append({"role": "user", "content": query})
    else:
        messages.append({"role": "user", "content": user_content})

    # Main Agentic Loop
    # Higher than a simple single-tool-call turn needs, since a research task may
    # legitimately take several rounds of search -> refine -> search again.
    max_iterations = 14
    iteration = 0
    # Track search_web usage across this turn so we can back off gracefully if the
    # network is unreachable, instead of letting the model retry it forever.
    search_round = 0
    consecutive_no_internet = 0
    any_tool_called = False
    nudged_no_tool_refusal = False
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Determine endpoint path
    # Ollama has standard chat completions at /v1/chat/completions (OpenAI compatible)
    # LM Studio also uses /v1/chat/completions
    url_suffix = "/v1/chat/completions"
    endpoint = f"{api_url.rstrip('/')}{url_suffix}"

    while iteration < max_iterations:
        iteration += 1
        print(f"Agent Loop iteration {iteration}/{max_iterations} | endpoint: {endpoint} | model: {model_name or '(auto)'}")

        # Call local LLM API
        # If model_name is empty, omit it — LM Studio uses whatever model is currently loaded
        payload = {
            "messages": messages,
            "stream": True
        }
        if model_name:
            payload["model"] = model_name
        # Only pass tools if the model seems to support it
        if openai_tools:
            payload["tools"] = openai_tools

        try:
            timeout = httpx.Timeout(connect=5.0, read=None, write=30.0, pool=10.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                # Quick connectivity check before streaming
                try:
                    probe = await client.get(f"{api_url.rstrip('/')}/v1/models", headers=headers, timeout=5.0)
                    print(f"LLM endpoint reachable, status: {probe.status_code}")
                except Exception as probe_err:
                    yield {"event": "error", "content": f"Cannot reach LLM endpoint at {api_url}.\nPlease ensure LM Studio is running and the local server is started (port 1234).\nError: {str(probe_err)}"}
                    return

                async with client.stream("POST", endpoint, json=payload, headers=headers) as response:
                    if response.status_code != 200:
                        err_text = await response.aread()
                        yield {"event": "error", "content": f"LLM API returned error status {response.status_code}: {err_text.decode('utf-8', errors='ignore')}"}
                        return

                    # Parse stream chunks
                    current_tool_calls = []
                    current_content = ""
                    current_images: List[Dict[str, str]] = []
                    is_thinking = False
                    thinking_content = ""
                    text_content = ""

                    # Helper to yield RAG search function inside tools
                    async def rag_search_func(q: str) -> str:
                        return await rag_store.query(
                            query_text=q,
                            top_k=4,
                            api_type=embedding_type,
                            api_url=embedding_url,
                            model_name=embedding_model,
                            api_key=api_key
                        )

                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                break
                            
                            try:
                                data = json.loads(data_str)
                                choice = data.get("choices", [{}])[0]
                                delta = choice.get("delta", {})

                                # 1. Handle Tool Calls Stream
                                tool_calls_delta = delta.get("tool_calls")
                                if tool_calls_delta:
                                    for tc in tool_calls_delta:
                                        idx = tc.get("index", 0)
                                        while len(current_tool_calls) <= idx:
                                            current_tool_calls.append({
                                                "id": "",
                                                "type": "function",
                                                "function": {"name": "", "arguments": ""}
                                            })
                                        
                                        # Merge delta properties
                                        if tc.get("id"):
                                            current_tool_calls[idx]["id"] += tc["id"]
                                        if tc.get("function", {}).get("name"):
                                            current_tool_calls[idx]["function"]["name"] += tc["function"]["name"]
                                        if tc.get("function", {}).get("arguments"):
                                            current_tool_calls[idx]["function"]["arguments"] += tc["function"]["arguments"]

                                # 2. Handle Image Output Stream
                                image_url = None
                                if delta.get("image_url"):
                                    image_field = delta["image_url"]
                                    if isinstance(image_field, dict):
                                        image_url = image_field.get("url")
                                    elif isinstance(image_field, str):
                                        image_url = image_field
                                elif delta.get("type") == "image_url" and isinstance(delta.get("url"), str):
                                    image_url = delta.get("url")

                                if image_url:
                                    image_obj = _parse_data_uri(image_url)
                                    if image_obj is None:
                                        image_obj = {"mime_type": "image/png", "url": image_url}
                                    current_images.append(image_obj)
                                    yield {"event": "image", "content": image_obj}

                                # 3. Handle Text Content Stream (with custom thinking parser)
                                content_delta = delta.get("content")
                                if content_delta:
                                    if isinstance(content_delta, list):
                                        # Flatten any structured content arrays
                                        for part in content_delta:
                                            if isinstance(part, dict) and part.get("type") == "image_url":
                                                image_url_part = part.get("image_url", {}).get("url") if isinstance(part.get("image_url"), dict) else None
                                                if image_url_part:
                                                    image_obj = _parse_data_uri(image_url_part) or {"mime_type": "image/png", "url": image_url_part}
                                                    current_images.append(image_obj)
                                                    yield {"event": "image", "content": image_obj}
                                            elif isinstance(part, str):
                                                content_delta = part
                                    elif isinstance(content_delta, dict) and content_delta.get("type") == "image_url":
                                        image_url_part = content_delta.get("image_url", {}).get("url") if isinstance(content_delta.get("image_url"), dict) else None
                                        if image_url_part:
                                            image_obj = _parse_data_uri(image_url_part) or {"mime_type": "image/png", "url": image_url_part}
                                            current_images.append(image_obj)
                                            yield {"event": "image", "content": image_obj}
                                        content_delta = None

                                if content_delta and isinstance(content_delta, str):
                                    current_content += content_delta
                                    
                                    # Parse thinking tags: <think> and </think>
                                    # If deepseek-r1 streams text, it outputs '<think>' then the reasoning, then '</think>'
                                    # We do a character-by-character search or simple substring check
                                    remaining = content_delta
                                    while remaining:
                                        if not is_thinking:
                                            # Look for <think> tag
                                            think_start = remaining.find("<think>")
                                            if think_start != -1:
                                                # Yield any text before <think>
                                                before = remaining[:think_start]
                                                if before:
                                                    text_content += before
                                                    yield {"event": "text", "content": before}
                                                
                                                is_thinking = True
                                                yield {"event": "thinking_start", "content": ""}
                                                remaining = remaining[think_start + 7:]
                                            else:
                                                # Just normal text
                                                text_content += remaining
                                                yield {"event": "text", "content": remaining}
                                                break
                                        else:
                                            # Look for </think> tag
                                            think_end = remaining.find("</think>")
                                            if think_end != -1:
                                                # Yield thinking content before </think>
                                                think_text = remaining[:think_end]
                                                if think_text:
                                                    thinking_content += think_text
                                                    yield {"event": "thinking", "content": think_text}
                                                
                                                is_thinking = False
                                                yield {"event": "thinking_end", "content": ""}
                                                remaining = remaining[think_end + 8:]
                                            else:
                                                # Still thinking
                                                thinking_content += remaining
                                                yield {"event": "thinking", "content": remaining}
                                                break
                            except Exception as e:
                                # Sometimes json parse fails for incomplete metadata, safe to ignore
                                pass

                    # Clean up tool calls structure
                    valid_tool_calls = []
                    for tc in current_tool_calls:
                        if tc["function"]["name"]:
                            # Parse arguments JSON safely
                            args_str = tc["function"]["arguments"]
                            try:
                                args_dict = json.loads(args_str) if args_str else {}
                            except Exception:
                                # Sometimes argument JSON is incomplete, try to clean it
                                try:
                                    # simple fixer for unclosed brackets
                                    args_dict = json.loads(args_str + "}")
                                except Exception:
                                    args_dict = {"raw_arguments": args_str}
                            
                            valid_tool_calls.append({
                                "id": tc["id"] or f"call_{iteration}_{len(valid_tool_calls)}",
                                "type": "function",
                                "function": {
                                    "name": tc["function"]["name"],
                                    "arguments": args_dict
                                }
                            })

                    # Add LLM response to history for next rounds
                    # `content` must always be present as a string (never omitted/null) —
                    # some OpenAI-compatible servers (e.g. LM Studio serving Gemma) reject
                    # a tool-call-only assistant message that lacks a `content` field.
                    assistant_msg = {"role": "assistant", "content": current_content or ""}
                    if valid_tool_calls:
                        # Per the OpenAI tool-calling spec, function.arguments must be a JSON
                        # *string* here, not the parsed object `valid_tool_calls` carries for
                        # execution/display — sending an object fails LM Studio's schema check.
                        assistant_msg["tool_calls"] = [
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["function"]["name"],
                                    "arguments": json.dumps(tc["function"]["arguments"])
                                }
                            }
                            for tc in valid_tool_calls
                        ]
                    messages.append(assistant_msg)

                    # If no tool calls requested, we are DONE! ...unless this looks like the
                    # model guessing it can't look something up rather than actually trying,
                    # while a search tool sits right there unused — force one corrective retry.
                    if not valid_tool_calls:
                        search_tool_available = bool(openai_tools) and any(
                            t.get("function", {}).get("name") == "search_web" for t in openai_tools
                        )
                        if (
                            not any_tool_called
                            and not nudged_no_tool_refusal
                            and search_tool_available
                            and _looks_like_capability_refusal(text_content)
                        ):
                            nudged_no_tool_refusal = True
                            yield {
                                "event": "text",
                                "content": "\n\n_(That sounded like a guess rather than an actual check — retrying with a live search instead.)_\n\n"
                            }
                            messages.append({
                                "role": "user",
                                "content": (
                                    "You do have a working search_web tool with real internet access right now. "
                                    "Actually call it to check this instead of assuming you can't — call search_web "
                                    "with a relevant query, then answer based on what it returns."
                                )
                            })
                            continue
                        yield {"event": "done", "content": text_content}
                        return

                    # 3. Execute Tool Calls
                    any_tool_called = True
                    for tool_call in valid_tool_calls:
                        tc_id = tool_call["id"]
                        tc_name = tool_call["function"]["name"]
                        tc_args = tool_call["function"]["arguments"]

                        tool_start_content = {
                            "id": tc_id,
                            "name": tc_name,
                            "arguments": tc_args
                        }
                        if tc_name == "search_web":
                            search_round += 1
                            tool_start_content["search_round"] = search_round

                        # Notify frontend that tool execution started
                        yield {
                            "event": "tool_start",
                            "content": tool_start_content
                        }

                        # Run the tool
                        tool_result = ""
                        try:
                            # Check if it is an MCP tool
                            if "__" in tc_name:
                                tool_result = await mcp_manager.call_mcp_tool(tc_name, tc_args)
                            else:
                                # Built-in tools
                                tool_result = await execute_tool(tc_name, tc_args, rag_search_func)
                        except Exception as e:
                            tool_result = f"Error executing tool: {str(e)}"

                        if tc_name == "search_web":
                            consecutive_no_internet = consecutive_no_internet + 1 if tool_result.startswith(NO_INTERNET_PREFIX) else 0

                        # Notify frontend that tool execution finished
                        yield {
                            "event": "tool_end",
                            "content": {
                                "id": tc_id,
                                "name": tc_name,
                                "output": tool_result
                            }
                        }

                        # Append tool response message to history
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "name": tc_name,
                            "content": tool_result
                        })

                    # Circuit breaker: if search_web has failed for lack of connectivity
                    # a couple of times in a row, stop offering it — otherwise the model
                    # tends to just keep retrying a search that can never succeed this turn.
                    if consecutive_no_internet >= 2 and openai_tools:
                        openai_tools = [t for t in openai_tools if t.get("function", {}).get("name") != "search_web"] or None
                        yield {
                            "event": "text",
                            "content": "\n\n_(No internet connection detected — I'll stop trying to search and answer with what I already know.)_\n\n"
                        }

        except Exception as e:
            yield {"event": "error", "content": f"Connection/Execution failed: {str(e)}\n{traceback.format_exc()}"}
            return

    yield {"event": "error", "content": "Agent exceeded maximum iteration depth without resolving."}
