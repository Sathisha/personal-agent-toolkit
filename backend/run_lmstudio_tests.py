import argparse
import json
import os
import sys
import time
import difflib
from pathlib import Path
from typing import Any, Dict, List, Optional
import html

import httpx


DEFAULT_PROMPTS = {
    "warmup": "Please say hello.",
    "latency": "What is the capital of France?",
    "keyword_quality": "Translate the following sentence into German: 'The quick brown fox jumped over the lazy dog.'",
    "instruction": "Summarize this sentence in one short sentence: 'The quick brown fox jumped over the lazy dog to visit the sleeping turtle by the river.'",
    "long_generation": "Give me a short history of the Internet in at least 120 words."
}

KEYWORD_EXPECTATIONS = {
    "keyword_quality": ["schnell", "braun", "fuchs", "Hund"],
    "instruction": ["fox", "lazy", "turtle", "river"]
}


def default_output_files(output_dir: Path) -> Dict[str, Path]:
    return {
        "json": output_dir / "lmstudio_model_test_results.json",
        "dashboard": output_dir / "lmstudio_model_test_dashboard.html"
    }


class LMStudioTester:
    def __init__(self, endpoint: str, control_endpoint: Optional[str] = None, api_key: str = "", timeout: int = 120):
        self.endpoint = endpoint.rstrip("/")
        self.control_endpoint = control_endpoint.rstrip("/") if control_endpoint else None
        self.api_key = api_key
        self.timeout = timeout
        self.client = httpx.Client(timeout=httpx.Timeout(timeout, connect=30))
        self.headers: Dict[str, str] = {}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

    def _log(self, msg: str) -> None:
        print(msg)

    def _build_url(self, path: str, base: Optional[str] = None) -> str:
        base_url = base.rstrip("/") if base else self.endpoint
        return f"{base_url}/{path.lstrip('/')}"

    def _extract_text(self, response: httpx.Response) -> str:
        data = response.json()
        if isinstance(data, dict) and "choices" in data and data["choices"]:
            first = data["choices"][0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    return message.get("content", "").strip()
                return first.get("text", "").strip()
        if isinstance(data, dict) and "output" in data:
            output = data["output"]
            if isinstance(output, str):
                return output.strip()
            if isinstance(output, list) and output:
                return str(output[0]).strip()
        if isinstance(data, dict) and "text" in data:
            return data["text"].strip()
        if isinstance(data, dict) and "message" in data:
            msg = data["message"]
            if isinstance(msg, str):
                return msg.strip()
        return json.dumps(data)

    def _send_request(self, model: str, prompt: str) -> Dict[str, Any]:
        payloads = [
            ("/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 256, "temperature": 0.2}),
            ("/v1/completions", {"model": model, "prompt": prompt, "max_tokens": 256, "temperature": 0.2}),
        ]
        last_error = None
        for path, json_body in payloads:
            url = self._build_url(path)
            try:
                response = self.client.post(url, json=json_body, headers=self.headers)
                if response.status_code == 200:
                    text = self._extract_text(response)
                    return {
                        "success": True,
                        "text": text,
                        "status_code": response.status_code,
                        "raw": response.json()
                    }
                last_error = f"{response.status_code}: {response.text}"
            except Exception as exc:
                last_error = str(exc)
        return {
            "success": False,
            "error": last_error,
            "status_code": None,
            "text": ""
        }

    def _measure_request(self, model: str, prompt: str) -> Dict[str, Any]:
        start = time.perf_counter()
        result = self._send_request(model, prompt)
        elapsed = (time.perf_counter() - start) * 1000.0
        result["latency_ms"] = elapsed
        return result

    def warmup(self, model: str, prompt: str) -> Dict[str, Any]:
        self._log(f"Warmup request for model '{model}'")
        return self._measure_request(model, prompt)

    def run_latency_test(self, model: str, prompt: str, samples: int = 3) -> Dict[str, Any]:
        self._log(f"Running latency test for model '{model}' ({samples} samples)")
        latencies = []
        texts = []
        errors = []
        for i in range(samples):
            result = self._measure_request(model, prompt)
            latencies.append(result.get("latency_ms", 0.0))
            texts.append(result.get("text", ""))
            if not result["success"]:
                errors.append(result.get("error"))
        latencies_sorted = sorted(latencies)
        return {
            "latency_ms": {
                "mean": sum(latencies) / len(latencies),
                "median": latencies_sorted[len(latencies_sorted) // 2],
                "p90": latencies_sorted[min(len(latencies_sorted) - 1, int(len(latencies_sorted) * 0.9))],
                "samples": latencies
            },
            "sample_texts": texts,
            "errors": errors,
            "success": len(errors) == 0
        }

    def run_repeatability_test(self, model: str, prompt: str, repeats: int = 3) -> Dict[str, Any]:
        self._log(f"Running repeatability test for model '{model}' ({repeats} repeats)")
        responses = []
        for i in range(repeats):
            result = self._measure_request(model, prompt)
            if not result["success"]:
                return {"success": False, "error": result.get("error"), "responses": responses}
            responses.append(result["text"])

        scores = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                seq = difflib.SequenceMatcher(a=responses[i], b=responses[j])
                scores.append(seq.ratio())

        return {
            "success": True,
            "responses": responses,
            "repeatability_score": round(sum(scores) / len(scores), 4) if scores else 1.0
        }

    def run_keyword_test(self, model: str, prompt: str, expected_keywords: List[str]) -> Dict[str, Any]:
        self._log(f"Running keyword match quality test for model '{model}'")
        result = self._measure_request(model, prompt)
        if not result["success"]:
            return {"success": False, "error": result.get("error"), "text": result.get("text", "")}
        text = result["text"].lower()
        hits = [keyword for keyword in expected_keywords if keyword.lower() in text]
        return {
            "success": True,
            "text": result["text"],
            "keyword_hits": hits,
            "expected_keywords": expected_keywords,
            "keyword_score": round(len(hits) / len(expected_keywords), 4)
        }

    def run_instruction_test(self, model: str, prompt: str) -> Dict[str, Any]:
        self._log(f"Running instruction-following test for model '{model}'")
        result = self._measure_request(model, prompt)
        if not result["success"]:
            return {"success": False, "error": result.get("error"), "text": result.get("text", "")}
        text = result["text"].strip()
        is_short = len(text.split()) <= 40
        return {
            "success": True,
            "text": text,
            "instruction_followed": is_short,
            "word_count": len(text.split())
        }

    def run_long_generation_test(self, model: str, prompt: str, min_words: int = 120) -> Dict[str, Any]:
        self._log(f"Running long generation test for model '{model}'")
        result = self._measure_request(model, prompt)
        if not result["success"]:
            return {"success": False, "error": result.get("error"), "text": result.get("text", "")}
        word_count = len(result["text"].split())
        return {
            "success": True,
            "text": result["text"],
            "word_count": word_count,
            "meets_length": word_count >= min_words
        }

    def _try_control_endpoint(self, path: str, method: str = "GET", payload: Optional[Dict[str, Any]] = None) -> Optional[httpx.Response]:
        if not self.control_endpoint:
            return None
        url = f"{self.control_endpoint.rstrip('/')}/{path.lstrip('/')}"
        try:
            response = self.client.request(method, url, json=payload, headers=self.headers)
            if response.status_code in (200, 201, 202, 204):
                return response
        except Exception:
            pass
        return None

    def get_loaded_models(self) -> List[str]:
        candidates = ["/v1/models", "/models", "/api/models"]
        for path in candidates:
            response = self._try_control_endpoint(path, "GET")
            if response is not None:
                try:
                    data = response.json()
                except Exception:
                    continue
                if isinstance(data, dict) and "models" in data:
                    if isinstance(data["models"], list):
                        return [m.get("id") or m.get("name") or str(m) for m in data["models"]]
                if isinstance(data, list):
                    return [m.get("id") or m.get("name") or str(m) for m in data if isinstance(m, dict)]
        return []

    def try_unload_model(self, model: str) -> bool:
        if not self.control_endpoint:
            return False
        paths = [f"/v1/models/{model}/unload", "/v1/models/unload", f"/models/{model}/unload", "/models/unload"]
        for path in paths:
            response = self._try_control_endpoint(path, "POST", {"model": model}) or self._try_control_endpoint(path, "POST", {})
            if response is not None:
                return True
        return False

    def try_load_model(self, model: str) -> bool:
        if not self.control_endpoint:
            return False
        paths = [f"/v1/models/{model}/load", "/v1/models/load", f"/models/{model}/load", "/models/load", "/v1/models", "/models"]
        for path in paths:
            payload = {"model": model}
            response = self._try_control_endpoint(path, "POST", payload)
            if response is not None:
                return True
        return False

    def maybe_prepare_model(self, model: str, switch_models: bool = False) -> Dict[str, Any]:
        if not self.control_endpoint:
            return {"prepared": False, "message": "Model control endpoint not configured."}

        loaded = self.get_loaded_models()
        message = []
        if switch_models and loaded:
            for loaded_model in loaded:
                if loaded_model != model:
                    if self.try_unload_model(loaded_model):
                        message.append(f"Unloaded {loaded_model}")
        if model not in loaded:
            if self.try_load_model(model):
                message.append(f"Loaded {model}")
        return {"prepared": bool(message), "message": "; ".join(message) if message else "No model control action taken."}

    def run_model_tests(self, model: str) -> Dict[str, Any]:
        model_results: Dict[str, Any] = {"model": model, "tests": {}}
        try:
            model_results["control"] = self.maybe_prepare_model(model, switch_models=True)
        except Exception as exc:
            model_results["control"] = {"prepared": False, "message": f"Control failed: {exc}"}

        first = self.warmup(model, DEFAULT_PROMPTS["warmup"])
        model_results["tests"]["warmup_first_load"] = {
            "success": first.get("success", False),
            "latency_ms": first.get("latency_ms"),
            "text": first.get("text"),
            "error": first.get("error")
        }

        second = self.warmup(model, DEFAULT_PROMPTS["warmup"])
        model_results["tests"]["warmup_after_first_load"] = {
            "success": second.get("success", False),
            "latency_ms": second.get("latency_ms"),
            "text": second.get("text"),
            "error": second.get("error")
        }

        latency = self.run_latency_test(model, DEFAULT_PROMPTS["latency"], samples=3)
        model_results["tests"]["latency"] = latency

        repeatability = self.run_repeatability_test(model, DEFAULT_PROMPTS["latency"], repeats=3)
        model_results["tests"]["repeatability"] = repeatability

        keyword_quality = self.run_keyword_test(model, DEFAULT_PROMPTS["keyword_quality"], KEYWORD_EXPECTATIONS["keyword_quality"])
        model_results["tests"]["keyword_quality"] = keyword_quality

        instruction = self.run_instruction_test(model, DEFAULT_PROMPTS["instruction"])
        model_results["tests"]["instruction_following"] = instruction

        long_gen = self.run_long_generation_test(model, DEFAULT_PROMPTS["long_generation"], min_words=120)
        model_results["tests"]["long_generation"] = long_gen

        return model_results


def render_dashboard(results: Dict[str, Any]) -> str:
    json_data = json.dumps(results, indent=2)
    table_rows = []
    for model_result in results["models"]:
        tests = model_result["tests"]
        latency = tests["latency"]["latency_ms"]
        repeatability = tests["repeatability"].get("repeatability_score") if tests["repeatability"].get("success") else None
        keyword_score = tests["keyword_quality"].get("keyword_score")
        long_ok = tests["long_generation"].get("meets_length")
        table_rows.append(
            f"<tr><td>{model_result['model']}</td><td>{latency['median']:.1f}</td><td>{latency['p90']:.1f}</td><td>{repeatability}</td><td>{keyword_score}</td><td>{'✅' if long_ok else '❌'}</td></tr>"
        )
    table_html = "\n".join(table_rows)
    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>LM Studio Model Comparison Dashboard</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.5; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
    th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
    th {{ background: #f4f4f4; }}
    pre {{ background: #111; color: #eee; padding: 16px; overflow-x: auto; border-radius: 8px; }}
    .status {{ margin-bottom: 16px; }}
  </style>
</head>
<body>
  <h1>LM Studio Model Comparison Dashboard</h1>
  <p class=\"status\">Report generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
  <h2>Summary</h2>
  <table>
    <thead>
      <tr><th>Model</th><th>Median Latency (ms)</th><th>P90 Latency (ms)</th><th>Repeatability</th><th>Keyword Score</th><th>Long Output</th></tr>
    </thead>
    <tbody>
      {table_html}
    </tbody>
  </table>
  <h2>JSON Results</h2>
  <pre>{html.escape(json_data)}</pre>
</body>
</html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Run locally deployed LM Studio model comparison tests.")
    parser.add_argument("--config", default=None, help="Optional JSON config file path")
    parser.add_argument("--endpoint", default=None, help="LM Studio API base URL, for example http://127.0.0.1:11434")
    parser.add_argument("--models", nargs="+", default=None, help="List of model names to compare")
    parser.add_argument("--control-endpoint", default=None, help="Optional LM Studio model control endpoint for load/unload actions")
    parser.add_argument("--api-key", default=None, help="Optional API key for LM Studio or local API authentication")
    parser.add_argument("--output-dir", default=".", help="Directory to save JSON and HTML results")
    parser.add_argument("--skip-control", action="store_true", help="Do not attempt model load/unload control calls")
    args = parser.parse_args()

    config_data: Dict[str, Any] = {}
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Config file not found: {config_path}")
            return 1
        with config_path.open("r", encoding="utf-8") as f:
            config_data = json.load(f)

    endpoint = args.endpoint or config_data.get("endpoint")
    models = args.models or config_data.get("models")
    control_endpoint = args.control_endpoint if args.control_endpoint is not None else config_data.get("control_endpoint")
    api_key = args.api_key if args.api_key is not None else config_data.get("api_key", "")
    output_dir_path = args.output_dir or config_data.get("output_dir", ".")

    if not endpoint or not models:
        print("Error: --endpoint and --models are required, either via CLI or config file.")
        return 1

    output_dir = Path(output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = default_output_files(output_dir)

    tester = LMStudioTester(
        endpoint=endpoint,
        control_endpoint=control_endpoint if not args.skip_control else None,
        api_key=api_key
    )

    results = {
        "endpoint": endpoint,
        "control_endpoint": control_endpoint,
        "api_key": api_key,
        "models": [],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "summary": {
            "tests": ["warmup_first_load", "warmup_after_first_load", "latency", "repeatability", "keyword_quality", "instruction_following", "long_generation"]
        }
    }

    for model in models:
        print(f"\n=== Testing model: {model} ===")
        model_result = tester.run_model_tests(model)
        results["models"].append(model_result)

    json_path = files["json"]
    dashboard_path = files["dashboard"]

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with dashboard_path.open("w", encoding="utf-8") as f:
        f.write(render_dashboard(results))

    print(f"\nResults saved to: {json_path.resolve()} and {dashboard_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
