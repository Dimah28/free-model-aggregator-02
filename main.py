#!/usr/bin/env python3
"""free-model-aggregator-02: one server, two wires, live free OpenRouter models.

The server reads its brief from $MAKEFUN_PROMPT_FILE (the free-model list
with the house ratings and the six prompts' manifest), registers its catalog
with the env at $MAKEFUN_ENV_URL, and answers OpenAI Chat Completions and
Anthropic Messages requests by dialling a free OpenRouter model through the
grader's gateway at $OPENAI_BASE_URL/chat/completions.

Routing: free models that meet the prompt's requirements, ordered by the
house rating (highest first, id ascending as tie-break), with failover to
the next candidate when the gateway refuses a call. Refusals are reported
on the x-makefun-upstream-failures header so the failover column can see
a refusal that was recovered from.

Standard library only. Deterministic: no wall clock or randomness reaches
any request or response; fixed creation timestamps, message ids taken from
the upstream answer.
"""

import json
import os
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_BODY = 8 * 1024 * 1024
FIXED_CREATED = 1767225600
FIXED_CREATED_AT = "2026-01-01T00:00:00Z"
GATEWAY_BASE = "https://openrouter.ai/api/v1"

# No proxy: the sandbox reaches only the env and the gateway.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


# ---------------------------------------------------------------- brief

def _scan_docs(text):
    """Yield every JSON object in `text` that carries a "schema" member."""
    decoder = json.JSONDecoder()
    docs = []
    pos = 0
    end = len(text)
    while pos < end:
        start = text.find("{", pos)
        if start < 0:
            break
        try:
            obj, stop = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            pos = start + 1
            continue
        if isinstance(obj, dict) and "schema" in obj:
            docs.append(obj)
        pos = stop
    return docs


def _load_brief():
    """Return (models, prompts) from the brief file, or ([], []) outside one."""
    path = os.environ.get("MAKEFUN_PROMPT_FILE", "")
    models = []
    prompts = []
    if not path:
        return models, prompts
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        return models, prompts
    for doc in _scan_docs(text):
        if isinstance(doc.get("models"), list):
            models = doc["models"]
        if isinstance(doc.get("prompts"), list):
            prompts = doc["prompts"]
    return models, prompts


def _normalise_model(raw):
    if not isinstance(raw, dict):
        return None
    model_id = raw.get("id")
    if not isinstance(model_id, str) or not model_id:
        return None
    rating = raw.get("rating_milli")
    if not isinstance(rating, (int, float)) or isinstance(rating, bool):
        rating = 0
    context = raw.get("context_length")
    if not isinstance(context, (int, float)) or isinstance(context, bool):
        context = None
    return {
        "id": model_id,
        "rating": rating,
        "context_length": context,
        "tools": bool(raw.get("tools")),
        "vision": bool(raw.get("vision")),
        "json": bool(raw.get("json")),
        "display_name": raw.get("display_name") or model_id,
    }


def _build_catalog(models):
    rows = []
    seen = set()
    for raw in models:
        model = _normalise_model(raw)
        if model is None:
            continue
        if raw.get("free") is not True:
            continue
        if model["id"] in seen:
            continue
        seen.add(model["id"])
        rows.append({
            "id": model["id"],
            "free": True,
            "context_length": model["context_length"],
        })
    if not rows:
        return {"providers": []}
    return {"providers": [{
        "id": "openrouter",
        "base_url": GATEWAY_BASE,
        "key_env": "OPENROUTER_API_KEY",
        "wire": "openai",
        "models": rows,
    }]}


_RAW_MODELS, _RAW_PROMPTS = _load_brief()
MODELS = []
for _raw in _RAW_MODELS:
    _model = _normalise_model(_raw)
    if _model is not None and _raw.get("free") is True:
        MODELS.append(_model)
PROMPTS = [p for p in _RAW_PROMPTS if isinstance(p, dict)]
CATALOG = _build_catalog(_RAW_MODELS)
CATALOG_JSON = json.dumps(CATALOG, separators=(",", ":"))

# Trial-local memory: models the gateway refused in this trial.
_REJECTED = set()
_REJECTED_LOCK = threading.Lock()

# Arrival order of completion requests, shared by both wires.
_PROMPT_INDEX = 0
_PROMPT_LOCK = threading.Lock()


def _next_prompt_index():
    global _PROMPT_INDEX
    with _PROMPT_LOCK:
        index = _PROMPT_INDEX
        _PROMPT_INDEX += 1
    return index


# ---------------------------------------------------------------- routing

def _candidates(requires):
    """Free models meeting `requires`, best rating first, id ascending."""
    req = requires or {}
    out = []
    for model in MODELS:
        ok = True
        for cap in ("tools", "vision", "json"):
            if req.get(cap) is True and not model[cap]:
                ok = False
                break
        if ok and req.get("min_context") is not None:
            ctx = model["context_length"]
            if ctx is None or ctx < req["min_context"]:
                ok = False
        if ok:
            out.append(model)
    out.sort(key=lambda m: (-m["rating"], m["id"]))
    return out


def _requires_from_body(body, wire):
    """Infer requirements from a request body when the manifest has none."""
    req = {}
    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        req["tools"] = True
    messages = body.get("messages") or []
    if wire == "openai":
        fmt = body.get("response_format")
        if isinstance(fmt, dict) and fmt.get("type") in ("json_object", "json_schema"):
            req["json"] = True
        for message in messages:
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        req["vision"] = True
    else:
        for message in messages:
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image":
                        req["vision"] = True
    return req


def _manifest_requires(index):
    if 0 <= index < len(PROMPTS):
        req = PROMPTS[index].get("requires")
        if isinstance(req, dict):
            return req
    return None


# ---------------------------------------------------------------- gateway

def _gateway_url():
    base = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
    if not base:
        return ""
    if not base.endswith("/v1"):
        base += "/v1"
    return base + "/chat/completions"


def _failure_row(status, raw, model_id):
    code = None
    try:
        parsed = json.loads(raw.decode("utf-8", "replace"))
        if isinstance(parsed, dict):
            err = parsed.get("error")
            if isinstance(err, dict):
                code = err.get("code") or err.get("type")
            if not code:
                code = parsed.get("code") or parsed.get("type")
    except Exception:
        code = None
    if not code:
        code = "http_%d" % status if status else "gateway_unreachable"
    return {"status": status, "code": str(code), "model": model_id}


def _dial_once(chat_body, model_id, stream, timeout):
    """One gateway call. Returns (response, None) or (None, (status, raw)).

    status 0 means the gateway could not be reached at all.
    """
    url = _gateway_url()
    body = dict(chat_body)
    body["model"] = model_id
    cap = body.get("max_tokens")
    if isinstance(cap, bool) or not isinstance(cap, int):
        if not isinstance(body.get("max_completion_tokens"), int):
            body["max_tokens"] = 1024
    elif cap > 8192:
        body["max_tokens"] = 8192
    elif cap < 1:
        body["max_tokens"] = 1024
    # The gateway refuses an output cap over 8192 under either name.
    alt = body.get("max_completion_tokens")
    if isinstance(alt, int) and not isinstance(alt, bool) and alt > 8192:
        body["max_completion_tokens"] = 8192
    body["stream"] = bool(stream)
    if stream:
        opts = body.get("stream_options")
        if not isinstance(opts, dict):
            body["stream_options"] = {"include_usage": True}
        elif "include_usage" not in opts:
            opts = dict(opts)
            opts["include_usage"] = True
            body["stream_options"] = opts
    data = json.dumps(body, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST")
    request.add_header("Content-Type", "application/json")
    request.add_header("Authorization", "Bearer " + os.environ.get("OPENAI_API_KEY", ""))
    try:
        return _OPENER.open(request, timeout=timeout), None
    except urllib.error.HTTPError as err:
        raw = b""
        try:
            raw = err.read()
        except Exception:
            raw = b""
        return None, (err.code, raw)
    except Exception as exc:  # gateway unreachable, timeout, DNS, ...
        return None, (0, str(exc).encode("utf-8", "replace"))


def _dial_with_failover(chat_body, stream, requires):
    """Try candidates in order (max 5, 28 s per prompt).

    Non-streaming: returns {"model", "body", "failures"} with the parsed
    upstream body. Streaming: returns {"model", "resp", "failures"} with
    the still-open upstream response. On failure: {"model": None, ...}.
    """
    deadline = time.monotonic() + 28.0
    ranked = _candidates(requires)
    with _REJECTED_LOCK:
        rejected = set(_REJECTED)
    ordered = [m for m in ranked if m["id"] not in rejected] or list(ranked)
    failures = []
    last_summary = "no candidate model"
    unreachable = False
    attempts = 0
    for model in ordered:
        if attempts >= 5:
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0.5:
            break
        attempts += 1
        resp, err = _dial_once(chat_body, model["id"], stream, min(25.0, remaining))
        if resp is not None:
            if stream:
                return {"model": model["id"], "resp": resp, "failures": failures}
            try:
                raw = resp.read()
            finally:
                resp.close()
            try:
                parsed = json.loads(raw.decode("utf-8"))
            except Exception:
                failures.append({"status": 200, "code": "invalid_response",
                                 "model": model["id"]})
                last_summary = "HTTP 200: upstream body was not JSON"
                continue
            return {"model": model["id"], "body": parsed, "failures": failures}
        status, raw = err
        if status == 0:
            unreachable = True
            failures.append({"status": 502, "code": "gateway_unreachable",
                             "model": model["id"]})
            last_summary = "gateway unreachable"
        else:
            failures.append(_failure_row(status, raw, model["id"]))
            with _REJECTED_LOCK:
                _REJECTED.add(model["id"])
            last_summary = "HTTP %d: %s" % (status, raw.decode("utf-8", "replace")[:500])
    return {
        "model": None,
        "failures": failures,
        "last_summary": last_summary,
        "unreachable": unreachable,
        "no_candidates": not ranked,
    }


# ---------------------------------------------------------------- wire helpers

def _openai_text(parsed):
    """Plain text of an OpenAI chat response, for usage synthesis."""
    try:
        choices = parsed.get("choices") or []
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                p.get("text", "") for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
    except Exception:
        pass
    return ""


def _synthesised_usage(text):
    completion = max(1, len(text) // 4)
    return {"prompt_tokens": 0, "completion_tokens": completion,
            "total_tokens": completion}


def _render_openai_json(parsed, model_id):
    if not isinstance(parsed, dict):
        parsed = {}
    parsed["model"] = model_id
    usage = parsed.get("usage")
    if not isinstance(usage, dict) or not isinstance(usage.get("completion_tokens"), int):
        parsed["usage"] = _synthesised_usage(_openai_text(parsed))
    return parsed


_STOP_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "refusal",
}


def _map_stop(finish_reason):
    return _STOP_MAP.get(finish_reason, "end_turn")


def _anthropic_to_chat(req):
    """Translate an Anthropic Messages body into a Chat Completions body."""
    chat = {}
    messages = []
    system = req.get("system")
    if isinstance(system, str) and system:
        messages.append({"role": "system", "content": system})
    elif isinstance(system, list):
        text = "".join(
            b.get("text", "") for b in system
            if isinstance(b, dict) and b.get("type") == "text"
        )
        if text:
            messages.append({"role": "system", "content": text})
    for message in req.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role", "user")
        content = message.get("content")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        blocks = content if isinstance(content, list) else []
        if role == "assistant":
            text_parts = []
            tool_calls = []
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    tool_calls.append({
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(
                                block.get("input") or {}, separators=(",", ":")),
                        },
                    })
            text = "".join(text_parts)
            out = {"role": "assistant",
                   "content": text if text else None}
            if tool_calls:
                out["tool_calls"] = tool_calls
            messages.append(out)
        else:
            text_parts = []
            image_parts = []

            def flush_text():
                if text_parts or image_parts:
                    parts = [{"type": "text", "text": t} for t in text_parts]
                    parts.extend(image_parts)
                    if len(parts) == 1 and parts[0]["type"] == "text":
                        messages.append({"role": "user",
                                         "content": parts[0]["text"]})
                    else:
                        messages.append({"role": "user", "content": parts})
                    del text_parts[:]
                    del image_parts[:]

            for block in blocks:
                if not isinstance(block, dict):
                    continue
                kind = block.get("type")
                if kind == "text":
                    text_parts.append(block.get("text", ""))
                elif kind == "image":
                    src = block.get("source") or {}
                    if src.get("type") == "base64":
                        image_parts.append({
                            "type": "image_url",
                            "image_url": {"url": "data:%s;base64,%s" % (
                                src.get("media_type", "image/png"),
                                src.get("data", ""))},
                        })
                    elif src.get("type") == "url":
                        image_parts.append({
                            "type": "image_url",
                            "image_url": {"url": src.get("url", "")},
                        })
                elif kind == "tool_result":
                    flush_text()
                    result = block.get("content")
                    if isinstance(result, list):
                        result = "".join(
                            x.get("text", "") for x in result
                            if isinstance(x, dict))
                    elif not isinstance(result, str):
                        result = json.dumps(result) if result is not None else ""
                    messages.append({
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id", ""),
                        "content": result,
                    })
            flush_text()
    chat["messages"] = messages
    tools = req.get("tools")
    if isinstance(tools, list) and tools:
        chat["tools"] = [{
            "type": "function",
            "function": {
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "parameters": t.get("input_schema") or {
                    "type": "object", "properties": {}},
            },
        } for t in tools if isinstance(t, dict)]
    choice = req.get("tool_choice")
    if isinstance(choice, dict):
        kind = choice.get("type")
        if kind == "auto":
            chat["tool_choice"] = "auto"
        elif kind == "any":
            chat["tool_choice"] = "required"
        elif kind == "tool":
            chat["tool_choice"] = {"type": "function",
                                   "function": {"name": choice.get("name", "")}}
        elif kind == "none":
            chat["tool_choice"] = "none"
    elif isinstance(choice, str):
        chat["tool_choice"] = {"auto": "auto", "any": "required",
                               "none": "none"}.get(choice, "auto")
    if isinstance(req.get("max_tokens"), int):
        chat["max_tokens"] = req["max_tokens"]
    for key in ("temperature", "top_p"):
        if key in req:
            chat[key] = req[key]
    stops = req.get("stop_sequences")
    if isinstance(stops, list) and stops:
        chat["stop"] = stops
    return chat


def _render_anthropic_json(parsed, model_id):
    choices = parsed.get("choices") if isinstance(parsed, dict) else None
    choice = choices[0] if isinstance(choices, list) and choices else {}
    upstream = choice.get("message") or {}
    finish = choice.get("finish_reason")
    content = []
    text = ""
    raw_content = upstream.get("content")
    if isinstance(raw_content, str):
        text = raw_content
    elif isinstance(raw_content, list):
        text = "".join(
            p.get("text", "") for p in raw_content
            if isinstance(p, dict) and p.get("type") == "text")
    if text:
        content.append({"type": "text", "text": text})
    for call in upstream.get("tool_calls") or []:
        fn = call.get("function") or {}
        try:
            parsed_input = json.loads(fn.get("arguments") or "{}")
        except Exception:
            parsed_input = {}
        if not isinstance(parsed_input, dict):
            parsed_input = {}
        content.append({
            "type": "tool_use",
            "id": call.get("id", ""),
            "name": fn.get("name", ""),
            "input": parsed_input,
        })
    usage = parsed.get("usage") if isinstance(parsed, dict) else None
    if isinstance(usage, dict) and isinstance(usage.get("completion_tokens"), int):
        in_tokens = usage.get("prompt_tokens", 0)
        out_tokens = usage["completion_tokens"]
    else:
        in_tokens = 0
        out_tokens = max(1, len(text) // 4)
    upstream_id = parsed.get("id", "") if isinstance(parsed, dict) else ""
    return {
        "id": "msg_" + str(upstream_id),
        "type": "message",
        "role": "assistant",
        "model": model_id,
        "content": content,
        "stop_reason": _map_stop(finish),
        "stop_sequence": None,
        "usage": {"input_tokens": in_tokens, "output_tokens": out_tokens},
    }


# ---------------------------------------------------------------- HTTP server

class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # -- plumbing ------------------------------------------------------

    def _failover_value(self, failures):
        rows = list(failures or [])[:8]
        value = json.dumps(rows, separators=(",", ":"))
        while len(value) > 8192 and rows:
            rows = rows[:-1]
            value = json.dumps(rows, separators=(",", ":"))
        return value if rows else None

    def _send_json(self, status, obj, failures=None):
        data = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        value = self._failover_value(failures)
        if value is not None:
            self.send_header("x-makefun-upstream-failures", value)
        self.end_headers()
        self.wfile.write(data)

    def _send_sse_start(self, failures=None):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        value = self._failover_value(failures)
        if value is not None:
            self.send_header("x-makefun-upstream-failures", value)
        self.end_headers()

    def _read_body(self):
        """Request body bytes, or None when it exceeds 8 MiB."""
        try:
            size = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            size = 0
        if size > MAX_BODY:
            return None
        if size <= 0:
            return b""
        return self.rfile.read(size)

    def _error_openai(self, status, message, err_type, failures=None):
        self._send_json(status, {"error": {"message": message, "type": err_type,
                                           "code": None}}, failures)

    def _error_anthropic(self, status, message, err_type, failures=None):
        self._send_json(status, {"type": "error",
                                 "error": {"type": err_type, "message": message}},
                        failures)

    def _parse_client_body(self, wire):
        """(body dict, None) or (None, error message)."""
        raw = self._read_body()
        if raw is None:
            self._error_openai(413, "request body over 8 MiB",
                               "invalid_request_error") if wire == "openai" \
                else self._error_anthropic(413, "request body over 8 MiB",
                                           "invalid_request_error")
            return None, "too large"
        try:
            body = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            message = "invalid JSON: %s" % exc
            if wire == "openai":
                self._error_openai(400, message, "invalid_request_error")
            else:
                self._error_anthropic(400, message, "invalid_request_error")
            return None, message
        if not isinstance(body, dict):
            message = "request body must be a JSON object"
            if wire == "openai":
                self._error_openai(400, message, "invalid_request_error")
            else:
                self._error_anthropic(400, message, "invalid_request_error")
            return None, message
        return body, None

    def _serve_failure(self, wire, result):
        failures = result.get("failures") or []
        if result.get("no_candidates"):
            message = "no free model meets this prompt's requirements"
            if wire == "openai":
                self._error_openai(503, message, "no_free_model", failures)
            else:
                self._error_anthropic(503, message, "overloaded_error", failures)
            return
        summary = result.get("last_summary") or "all gateway attempts failed"
        message = "upstream failed: %s" % summary
        if wire == "openai":
            self._error_openai(502, message, "api_error", failures)
        else:
            self._error_anthropic(502, message, "api_error", failures)

    # -- routes ----------------------------------------------------------

    def do_GET(self):
        try:
            path = self.path.split("?", 1)[0]
            if path == "/v1/models":
                self._handle_models()
            else:
                self._send_json(404, {"error": {"message": "not found",
                                                "type": "invalid_request_error",
                                                "code": None}})
        except Exception:
            traceback.print_exc()
            try:
                self._send_json(500, {"error": {"message": "internal error",
                                                "type": "api_error", "code": None}})
            except Exception:
                pass

    def do_POST(self):
        try:
            path = self.path.split("?", 1)[0]
            if path == "/v1/chat/completions":
                self._handle_openai()
            elif path == "/v1/messages":
                self._handle_anthropic()
            else:
                self._send_json(404, {"error": {"message": "not found",
                                                "type": "invalid_request_error",
                                                "code": None}})
        except Exception:
            traceback.print_exc()
            try:
                self._send_json(500, {"error": {"message": "internal error",
                                                "type": "api_error", "code": None}})
            except Exception:
                pass

    def _handle_models(self):
        ids = [m["id"] for m in MODELS]
        if self.headers.get("x-api-key") is not None:
            data = [{
                "type": "model",
                "id": model["id"],
                "display_name": model["display_name"],
                "created_at": FIXED_CREATED_AT,
            } for model in MODELS]
            self._send_json(200, {"data": data, "has_more": False})
        else:
            data = [{
                "id": model["id"],
                "object": "model",
                "created": FIXED_CREATED,
                "owned_by": "openrouter",
            } for model in MODELS]
            self._send_json(200, {"object": "list", "data": data})

    def _handle_openai(self):
        body, error = self._parse_client_body("openai")
        if error is not None:
            return
        index = _next_prompt_index()
        requires = _manifest_requires(index)
        if requires is None:
            requires = _requires_from_body(body, "openai")
        stream = bool(body.get("stream"))
        result = _dial_with_failover(body, stream, requires)
        if result.get("model") is None:
            self._serve_failure("openai", result)
            return
        if stream:
            self._send_sse_start(result["failures"])
            try:
                self._forward_openai_stream(result["resp"], result["model"])
            finally:
                try:
                    result["resp"].close()
                except Exception:
                    pass
        else:
            rendered = _render_openai_json(result["body"], result["model"])
            self._send_json(200, rendered, result["failures"])

    def _handle_anthropic(self):
        body, error = self._parse_client_body("anthropic")
        if error is not None:
            return
        index = _next_prompt_index()
        requires = _manifest_requires(index)
        if requires is None:
            requires = _requires_from_body(body, "anthropic")
        stream = bool(body.get("stream"))
        chat = _anthropic_to_chat(body)
        result = _dial_with_failover(chat, stream, requires)
        if result.get("model") is None:
            self._serve_failure("anthropic", result)
            return
        if stream:
            self._send_sse_start(result["failures"])
            try:
                self._forward_anthropic_stream(result["resp"], result["model"])
            finally:
                try:
                    result["resp"].close()
                except Exception:
                    pass
        else:
            rendered = _render_anthropic_json(result["body"], result["model"])
            self._send_json(200, rendered, result["failures"])

    # -- streaming -------------------------------------------------------

    def _forward_openai_stream(self, resp, model_id):
        while True:
            line = resp.readline()
            if not line:
                break
            if line.startswith(b"data:"):
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    self.wfile.write(line)
                    self.wfile.flush()
                    continue
                try:
                    chunk = json.loads(payload.decode("utf-8"))
                except Exception:
                    self.wfile.write(line)
                    self.wfile.flush()
                    continue
                if isinstance(chunk, dict):
                    chunk["model"] = model_id
                    out = ("data: " + json.dumps(chunk, separators=(",", ":"))
                           + "\n").encode("utf-8")
                    self.wfile.write(out)
                else:
                    self.wfile.write(line)
                self.wfile.flush()
            else:
                self.wfile.write(line)
                self.wfile.flush()

    def _emit_anthropic_event(self, event_type, payload):
        data = ("event: " + event_type + "\ndata: "
                + json.dumps(payload, separators=(",", ":")) + "\n\n")
        self.wfile.write(data.encode("utf-8"))
        self.wfile.flush()

    def _forward_anthropic_stream(self, resp, model_id):
        msg_id = None
        started = False
        text_open = False
        text_index = 0
        tool_blocks = {}   # upstream tool index -> {"index", "id", "name"}
        next_index = 0
        open_blocks = []
        out_tokens = 1
        finish_reason = None

        def close_blocks():
            for index in reversed(open_blocks):
                self._emit_anthropic_event(
                    "content_block_stop", {"type": "content_block_stop",
                                           "index": index})
            del open_blocks[:]

        def finish():
            nonlocal text_open
            close_blocks()
            self._emit_anthropic_event("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": _map_stop(finish_reason),
                          "stop_sequence": None},
                "usage": {"output_tokens": out_tokens},
            })
            self._emit_anthropic_event("message_stop", {"type": "message_stop"})
            text_open = False

        while True:
            line = resp.readline()
            if not line:
                if started:
                    finish()
                return
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                if started:
                    finish()
                return
            try:
                chunk = json.loads(payload.decode("utf-8"))
            except Exception:
                continue
            if not isinstance(chunk, dict):
                continue
            if not started:
                started = True
                upstream_id = chunk.get("id", "")
                msg_id = "msg_" + str(upstream_id)
                self._emit_anthropic_event("message_start", {
                    "type": "message_start",
                    "message": {
                        "id": msg_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": model_id,
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    },
                })
            choices = chunk.get("choices") or []
            choice = choices[0] if choices else {}
            if isinstance(choice, dict) and choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
            usage = chunk.get("usage")
            if isinstance(usage, dict) and isinstance(
                    usage.get("completion_tokens"), int):
                out_tokens = usage["completion_tokens"]
            delta = choice.get("delta") if isinstance(choice, dict) else None
            if not isinstance(delta, dict):
                continue
            text = delta.get("content")
            if isinstance(text, str) and text:
                if not text_open:
                    text_index = next_index
                    next_index += 1
                    self._emit_anthropic_event("content_block_start", {
                        "type": "content_block_start",
                        "index": text_index,
                        "content_block": {"type": "text", "text": ""},
                    })
                    open_blocks.append(text_index)
                    text_open = True
                self._emit_anthropic_event("content_block_delta", {
                    "type": "content_block_delta",
                    "index": text_index,
                    "delta": {"type": "text_delta", "text": text},
                })
            for call in delta.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                call_index = call.get("index", 0)
                block = tool_blocks.get(call_index)
                if block is None:
                    block_index = next_index
                    next_index += 1
                    fn = call.get("function") or {}
                    block = {
                        "index": block_index,
                        "id": call.get("id", ""),
                        "name": fn.get("name", ""),
                    }
                    tool_blocks[call_index] = block
                    self._emit_anthropic_event("content_block_start", {
                        "type": "content_block_start",
                        "index": block_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": block["id"],
                            "name": block["name"],
                            "input": {},
                        },
                    })
                    open_blocks.append(block_index)
                fragment = (call.get("function") or {}).get("arguments")
                if isinstance(fragment, str) and fragment:
                    self._emit_anthropic_event("content_block_delta", {
                        "type": "content_block_delta",
                        "index": block["index"],
                        "delta": {"type": "input_json_delta",
                                  "partial_json": fragment},
                    })


# ---------------------------------------------------------------- trial loop

def _env_request(method, path, payload=None):
    env_url = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
    if not env_url:
        return None
    data = None
    request = urllib.request.Request(env_url + path, method=method)
    if payload is not None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(env_url + path, data=data, method=method)
        request.add_header("Content-Type", "application/json")
    with _OPENER.open(request, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _register_and_poll(server, port):
    try:
        answer = _env_request("POST", "/register", {
            "protocol_version": 1,
            "port": port,
            "catalog": CATALOG,
        })
        sys.stderr.write("register: %s\n" % (answer,))
    except Exception:
        traceback.print_exc()
    while True:
        try:
            state = _env_request("GET", "/work")
            if isinstance(state, dict) and state.get("done") is True:
                break
        except Exception:
            pass
        time.sleep(0.1)
    server.shutdown()


def main(argv):
    if "--list" in argv:
        sys.stdout.write(CATALOG_JSON + "\n")
        return 0
    server = ThreadingHTTPServer(("0.0.0.0", 0), _Handler)
    server.daemon_threads = True
    port = server.server_address[1]
    sys.stderr.write("listening on 0.0.0.0:%d\n" % port)
    poller = threading.Thread(target=_register_and_poll, args=(server, port),
                              daemon=True)
    poller.start()
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
