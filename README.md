# Personal Agent Toolkit

Personal Agent Toolkit is the repo behind **Personal Agent** — a self-hosted, batteries-included chat UI and agent runtime that sits in front of **any OpenAI-compatible model API** — local (LM Studio, Ollama) or cloud (OpenAI-compatible endpoints). Point it at a model and you get an agentic tool-calling loop, live web search, document RAG, MCP server extensibility, image upload/paste, and voice I/O — none of which the default chat UI in LM Studio or Ollama gives you.

It runs entirely on your own machine via Docker Compose: your prompts, documents, and search queries never have to leave your network unless you choose to point it at a cloud model.

## Why not just use LM Studio's or Ollama's built-in chat?

LM Studio and Ollama are excellent at *serving* models, but their built-in chat windows are single-turn conversations with no ability to act. Personal Agent wraps the same models (or a cloud model, interchangeably) in an actual agent:

| | LM Studio / Ollama chat | Personal Agent |
|---|---|---|
| Talk to a local model | ✅ | ✅ |
| Talk to a cloud/OpenAI-compatible model | ❌ (separate tool) | ✅ — same UI, just change the endpoint |
| Tool-calling / agentic loop (read/write files, run shell commands, execute Python) | ❌ | ✅ |
| Live web search, with automatic query refinement until it finds a good answer | ❌ | ✅ (self-hosted, no API key) |
| RAG over your own documents (PDF/TXT/MD/code) | ❌ | ✅ |
| MCP server support (Postgres, GitHub, Memory, or any custom MCP server) | ❌ | ✅ |
| Image upload / clipboard screenshot paste for vision models | Partial | ✅ |
| Voice input (speech-to-text) and spoken responses (text-to-speech) | ❌ | ✅ |
| Visible "thinking" trace for reasoning models (e.g. DeepSeek R1 style) | ❌ | ✅ |
| Markdown + LaTeX math rendering in responses | ❌ | ✅ |
| Stop/cancel an in-flight response | Varies | ✅ |
| Graceful offline handling (won't hang or spam retries with no internet) | N/A | ✅ |

In short: LM Studio/Ollama answer "what does the model say?" — Personal Agent answers "what can the model *do*, using this model or that one, local or cloud, interchangeably?"

## Supported model backends

Configured per-session in the Settings panel — no restart needed to switch:

- **LM Studio** (local) — point at its OpenAI-compatible server (default `http://localhost:1234`)
- **Ollama** (local) — point at its OpenAI-compatible endpoint
- **OpenAI-compatible cloud endpoint** — any hosted API that speaks the `/v1/chat/completions` protocol (OpenAI itself, or any compatible proxy/gateway)

The same applies separately to the embedding backend used for RAG.

## Features

- **Agentic tool loop** — the model can call `read_file`, `write_file`, `run_command`, `python_interpreter`, `search_web`, and `query_documents`, looping (up to several rounds) until it has enough to answer, not just a single request/response.
- **Live web search** — backed by a self-hosted [SearXNG](https://github.com/searxng/searxng) instance (no API keys, no accounts). Multi-round research is shown in the chat as distinct "Search #1", "Search #2"... steps rather than a black box, and a no-internet condition is detected and reported cleanly instead of retrying forever.
- **RAG document library** — upload PDF/TXT/MD/code files, they're chunked and embedded in the background, and the agent can semantically search them on demand.
- **MCP servers** — connect any stdio-based [Model Context Protocol](https://modelcontextprotocol.io/) server (Postgres, GitHub, Memory, etc.) from the UI and its tools become available to the agent alongside the built-ins.
- **Vision** — attach images via file picker or paste a screenshot directly into the chat box.
- **Voice** — speech-to-text input and text-to-speech playback of responses, using the browser's built-in speech APIs.
- **Stop button** — cancel a response mid-stream.
- **Reasoning visibility** — `<think>...</think>` traces from reasoning models are shown in a collapsible "thinking" panel, separate from the final answer.

## Quick start (Docker Compose)

```bash
docker-compose build
docker-compose up
```

This brings up three containers:

- **backend** — FastAPI + WebSocket agent runtime, exposed on `localhost:8005`
- **frontend** — the chat UI, exposed on `localhost:3005`
- **searxng** — self-hosted search backend, internal-only (not exposed to the host)

Open `http://localhost:3005`, go to **Settings**, point the API connection at your running LM Studio/Ollama instance (or a cloud endpoint) and pick a model.

> **Note:** local backends (LM Studio/Ollama) running on your host machine are reached from inside the backend container via `host.docker.internal` — this is pre-wired in `docker-compose.yml`.

## Configuration

All runtime configuration (endpoint, model, API key, system prompt, embedding settings) lives in the Settings panel and is saved to your browser's `localStorage` — no `.env` editing required for normal use. Backend-only settings (workspace directory, SearXNG URL) are set via environment variables in `docker-compose.yml`.

## LM Studio model comparison harness

A separate utility script for benchmarking multiple locally loaded LM Studio models against each other (latency, repeatability, keyword-based quality, instruction-following).

### Files

- `backend/run_lmstudio_tests.py` — test harness for model comparison
- `backend/lmstudio_model_test_config.json` — sample test configuration

### Checks performed per model

1. Warmup / first-load latency
2. Steady-state latency
3. Repeatability consistency
4. Keyword-based quality
5. Instruction-following and long generation

### Running it

```bash
python backend/run_lmstudio_tests.py --endpoint http://127.0.0.1:11434 --models model-one model-two model-three --output-dir backend/results
```

If your LM Studio setup supports model load/unload control, include:

```bash
python backend/run_lmstudio_tests.py --endpoint http://127.0.0.1:11434 --control-endpoint http://127.0.0.1:11434 --api-key YOUR_API_KEY --models model-one model-two model-three --output-dir backend/results
```

You can also pass `--config backend/lmstudio_model_test_config.json` to load the endpoint, models, output directory, and `api_key` from a JSON file instead of flags.

Notes:

- The script issues an initial warmup request to reduce the impact of cold start.
- It optionally tries to unload other models and load the target model before testing, if a control endpoint is available.
- Outputs are written to `backend/results/lmstudio_model_test_results.json` and `backend/results/lmstudio_model_test_dashboard.html`.
- To run it against the containers, open a shell in the `backend` service and execute the script there.

Update `backend/lmstudio_model_test_config.json` to set your default endpoint and model names.
