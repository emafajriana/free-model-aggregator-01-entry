#!/usr/bin/env python3
"""free-model-aggregator-01: one server, two client APIs, ten free-tier providers.

The server reads the trial brief at $MAKEFUN_PROMPT_FILE (the provider
directory, the quality table and the workload manifest), probes every
provider's catalog route in that provider's own wire, builds one catalog
document, registers it with the env at $MAKEFUN_ENV_URL, and then answers
the env client's requests:

  POST /v1/chat/completions   OpenAI Chat Completions, JSON or SSE
  POST /v1/messages           Anthropic Messages, JSON or SSE
  GET  /v1/models             OpenAI list for Authorization: Bearer,
                              Anthropic list for x-api-key

Each client request names model "auto"; the server routes it to the
highest-rated free model that meets the request's requirements (from the
workload manifest, in arrival order), speaking that provider's wire
(openai, anthropic or gemini), and fails over to the next candidate when
a provider refuses (402, 429, 5xx or a transport error). Requests are
translated between wires field by field so that everything a provider
can observe -- system prompt, user text, images, tool names, max_tokens,
stop sequences and JSON mode -- arrives exactly as the client sent it,
and the provider's answer (text, usage, stop reason) is translated back
into the client's wire, streamed or not as asked.

`python3 main.py --list` prints the catalog document and exits 0; the
register body carries that same document. Everything is a deterministic
function of the brief, the requests and the providers' answers: no wall
clock, no randomness. Standard library only.
"""

import json
import os
import sys
import threading
import time
from http.client import HTTPConnection, HTTPSConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

PROTOCOL_VERSION = 1
PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"
ANTHROPIC_VERSION = "2023-06-01"
FIXED_CREATED = 1767225600
FIXED_CREATED_AT = "2026-01-01T00:00:00Z"
PROBE_TIMEOUT = 5.0
UPSTREAM_TIMEOUT = 20.0
POLL_INTERVAL = 0.1

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")


def log(message):
    sys.stderr.write("aggregator: %s\n" % message)
    sys.stderr.flush()


def is_record(value):
    return isinstance(value, dict)


# --------------------------------------------------------------------------
# The brief: three JSON documents found by their schema member.
# --------------------------------------------------------------------------

def _object_end(text, start):
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def documents_in(text):
    out = []
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
            except ValueError:
                value = None
            if is_record(value) and isinstance(value.get("schema"), str):
                out.append(value)
        start = text.find("{", start + 1)
    return out


class Provider:
    def __init__(self, row):
        self.id = row.get("id", "")
        self.name = row.get("name") or self.id
        self.wire = row.get("wire", "openai")
        self.auth = row.get("auth", "bearer")
        base = row.get("path") or row.get("base_path") or row.get("base_url") or row.get("base") or ""
        self.base = base if isinstance(base, str) else ""
        self.key_env = row.get("key_env", "") if isinstance(row.get("key_env", ""), str) else ""
        self.key = row.get("key", "") if isinstance(row.get("key", ""), str) else ""

    def dial_base(self):
        base = self.base.strip()
        if base.startswith("http://") or base.startswith("https://"):
            return base.rstrip("/")
        if base == "":
            return ENV_URL
        return ENV_URL + ("" if base.startswith("/") else "/") + base.rstrip("/")


class Brief:
    def __init__(self):
        self.providers = []
        self.ratings = {}          # (provider_id, model_id) -> row dict
        self.manifest = []         # workload request dicts in order
        self.error = None


def read_brief(path):
    brief = Brief()
    if not path:
        brief.error = "MAKEFUN_PROMPT_FILE is not set"
        return brief
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError as err:
        brief.error = "cannot read %s: %s" % (path, err)
        return brief
    docs = documents_in(text)
    directory = None
    ratings_doc = None
    workload_doc = None
    for doc in docs:
        if doc.get("schema") == PROVIDERS_SCHEMA and isinstance(doc.get("providers"), list):
            directory = doc
        elif doc.get("schema") == RATINGS_SCHEMA and isinstance(doc.get("ratings"), list):
            ratings_doc = doc
        elif doc.get("schema") == WORKLOAD_SCHEMA and isinstance(doc.get("manifest"), list):
            workload_doc = doc
    if directory is None:
        brief.error = "the brief holds no provider directory"
        return brief
    for row in directory["providers"]:
        if is_record(row) and isinstance(row.get("id"), str) and row["id"]:
            brief.providers.append(Provider(row))
    if ratings_doc is not None:
        for row in ratings_doc["ratings"]:
            if is_record(row) and isinstance(row.get("model"), str) and isinstance(row.get("provider"), str):
                brief.ratings[(row["provider"], row["model"])] = row
    if workload_doc is not None:
        for row in workload_doc["manifest"]:
            if is_record(row) and isinstance(row.get("id"), str):
                requires = row.get("requires") if is_record(row.get("requires")) else {}
                brief.manifest.append({
                    "id": row["id"],
                    "api": "anthropic" if row.get("api") == "anthropic" else "openai",
                    "stream": row.get("stream") is True,
                    "requires": requires,
                })
    return brief


BRIEF = read_brief(PROMPT_FILE)


# --------------------------------------------------------------------------
# HTTP plumbing (upstream calls to the env's providers).
# --------------------------------------------------------------------------

def http_request(method, url, body=None, headers=None, timeout=UPSTREAM_TIMEOUT):
    """One HTTP request; returns (status, headers dict, body bytes).

    A transport failure returns status 0 so callers can fail over.
    """
    parts = urlsplit(url)
    if parts.scheme == "https":
        conn = HTTPSConnection(parts.hostname, parts.port or 443, timeout=timeout)
    else:
        conn = HTTPConnection(parts.hostname, parts.port or 80, timeout=timeout)
    path = parts.path or "/"
    if parts.query:
        path += "?" + parts.query
    payload = None
    send_headers = dict(headers or {})
    if body is not None:
        payload = body if isinstance(body, bytes) else body.encode("utf-8")
        send_headers.setdefault("content-type", "application/json")
    try:
        conn.request(method, path, body=payload, headers=send_headers)
        response = conn.getresponse()
        data = response.read()
        return response.status, {k.lower(): v for k, v in response.getheaders()}, data
    except Exception:
        return 0, {}, b""
    finally:
        conn.close()


def auth_headers(provider):
    if not provider.key:
        return {}
    if provider.auth == "bearer":
        return {"authorization": "Bearer " + provider.key}
    if provider.auth == "x-api-key":
        return {"x-api-key": provider.key}
    if provider.auth == "x-goog-api-key":
        return {"x-goog-api-key": provider.key}
    return {}


# --------------------------------------------------------------------------
# Catalog probe and the catalog document.
# --------------------------------------------------------------------------

class ProbedModel:
    def __init__(self, model_id):
        self.id = model_id
        self.display_name = model_id
        self.free = None
        self.context_length = None
        self.tools = None
        self.vision = None
        self.json = None


def _bool_or_none(value):
    return value if isinstance(value, bool) else None


def _num_or_none(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    return None


def _zero_priced(row):
    pricing = row.get("pricing")
    if not is_record(pricing):
        return None
    values = []
    for item in pricing.values():
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            values.append(float(item))
        elif isinstance(item, str) and item.strip():
            try:
                values.append(float(item))
            except ValueError:
                pass
    if not values:
        return None
    return all(v == 0 for v in values)


def parse_listing(wire, payload):
    models = []
    if not is_record(payload):
        return models
    rows = payload.get("models") if wire == "gemini" else payload.get("data")
    if not isinstance(rows, list):
        return models
    for row in rows:
        if not is_record(row):
            continue
        if wire == "gemini":
            name = row.get("name")
            model_id = name[len("models/"):] if isinstance(name, str) and name.startswith("models/") else name
        else:
            model_id = row.get("id")
        if not isinstance(model_id, str) or not model_id:
            continue
        model = ProbedModel(model_id)
        display = row.get("display_name") or row.get("displayName")
        if isinstance(display, str) and display:
            model.display_name = display
        free = _bool_or_none(row.get("free"))
        model.free = free if free is not None else _zero_priced(row)
        model.context_length = (_num_or_none(row.get("context_length"))
                                or _num_or_none(row.get("context_window"))
                                or _num_or_none(row.get("inputTokenLimit")))
        model.tools = _bool_or_none(row.get("tools"))
        model.vision = _bool_or_none(row.get("vision"))
        model.json = _bool_or_none(row.get("json"))
        models.append(model)
    return models


def probe_provider(provider):
    base = provider.dial_base()
    url = base + ("/v1beta/models" if provider.wire == "gemini" else "/v1/models")
    headers = {"accept": "application/json"}
    headers.update(auth_headers(provider))
    if provider.wire == "anthropic":
        headers["anthropic-version"] = ANTHROPIC_VERSION
    status, _headers, data = http_request("GET", url, headers=headers, timeout=PROBE_TIMEOUT)
    if status < 200 or status >= 300:
        return False, []
    try:
        payload = json.loads(data.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return False, []
    return True, parse_listing(provider.wire, payload)


class Candidate:
    """One routable (provider, model) pair with its merged capabilities."""

    def __init__(self, provider, probed):
        self.provider = provider
        self.model_id = probed.id
        self.display_name = probed.display_name
        row = BRIEF.ratings.get((provider.id, probed.id))
        self.row = row
        if row is not None:
            self.free = row.get("free") is True
            self.context_length = row.get("context_length")
            self.tools = row.get("tools") is True
            self.vision = row.get("vision") is True
            self.json = row.get("json") is True
            rating = row.get("rating_milli")
            self.rating = rating if isinstance(rating, (int, float)) else 0
        else:
            self.free = probed.free is True
            self.context_length = probed.context_length
            self.tools = probed.tools is True
            self.vision = probed.vision is True
            self.json = probed.json is True
            self.rating = 0

    def catalog_model(self):
        return {"id": self.model_id, "free": self.free, "context_length": self.context_length}


PROBED = {}   # provider id -> (ok, [ProbedModel])
CANDIDATES = []  # every probed model, as Candidate
CATALOG = {"providers": []}


def build_catalog():
    """Probe every provider once and build the one catalog document."""
    global CATALOG
    providers_out = []
    for provider in BRIEF.providers:
        if ENV_URL:
            ok, models = probe_provider(provider)
        else:
            # No env to probe (an offline --list): fall back to the brief's
            # own quality table, which lists the free models per provider.
            ok = True
            models = []
            for (pid, mid), row in sorted(BRIEF.ratings.items()):
                if pid != provider.id:
                    continue
                model = ProbedModel(mid)
                model.free = row.get("free") is True
                model.context_length = row.get("context_length")
                model.tools = row.get("tools") is True
                model.vision = row.get("vision") is True
                model.json = row.get("json") is True
                models.append(model)
        PROBED[provider.id] = (ok, models)
        if not ok:
            log("probe failed for provider %s" % provider.id)
            continue
        seen = set()
        catalog_models = []
        for probed in sorted(models, key=lambda m: m.id):
            if probed.id in seen:
                continue
            seen.add(probed.id)
            candidate = Candidate(provider, probed)
            CANDIDATES.append(candidate)
            catalog_models.append(candidate.catalog_model())
        providers_out.append({
            "id": provider.id,
            "base_url": provider.base,
            "key_env": provider.key_env,
            "wire": provider.wire,
            "models": catalog_models,
        })
    CATALOG = {"providers": providers_out}
    return CATALOG


def served_model_ids():
    ids = []
    seen = set()
    for provider in CATALOG["providers"]:
        for model in provider["models"]:
            if model["free"] is True and model["id"] not in seen:
                seen.add(model["id"])
                ids.append(model["id"])
    return ids


# --------------------------------------------------------------------------
# Requirement handling and routing.
# --------------------------------------------------------------------------

def body_requirements(body, api):
    requires = {}
    messages = body.get("messages")
    if isinstance(messages, list):
        vision = False
        for message in messages:
            if not is_record(message):
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not is_record(block):
                    continue
                if block.get("type") in ("image", "image_url"):
                    vision = True
        if vision:
            requires["vision"] = True
    if isinstance(body.get("tools"), list) and body["tools"]:
        requires["tools"] = True
    if api == "openai":
        fmt = body.get("response_format")
        if is_record(fmt) and fmt.get("type") in ("json_object", "json_schema"):
            requires["json"] = True
    return requires


def meets(candidate, requires):
    if requires.get("tools") is True and not candidate.tools:
        return False
    if requires.get("vision") is True and not candidate.vision:
        return False
    if requires.get("json") is True and not candidate.json:
        return False
    min_context = requires.get("min_context")
    if isinstance(min_context, (int, float)):
        if candidate.context_length is None or candidate.context_length < min_context:
            return False
    return True


def rank_candidates(requires, client_api, client_body):
    """Free candidates meeting the requirements, best rating first.

    Routing policy: budgets and failure schedules are drawn per trial and
    never published, so the server cannot plan around them; it simply
    asks the best legal model first for every request and fails over in
    rating order when a provider refuses. A refusal on a provider's
    schedule costs one call but leaves the provider usable for the next
    request, which is why a refused provider is never blacklisted: the
    next request tries the best model first again.

    An OpenAI JSON-mode request can only ride a wire that carries JSON
    mode: the anthropic wire's providers always report it absent, so
    anthropic-wire candidates are excluded for those requests.
    """
    json_mode = False
    if client_api == "openai":
        fmt = client_body.get("response_format")
        json_mode = is_record(fmt) and fmt.get("type") in ("json_object", "json_schema")
    provider_order = {p.id: i for i, p in enumerate(BRIEF.providers)}
    chosen = []
    for candidate in CANDIDATES:
        if not candidate.free:
            continue
        if not meets(candidate, requires):
            continue
        if json_mode and candidate.provider.wire == "anthropic":
            continue
        chosen.append(candidate)
    chosen.sort(key=lambda c: (-c.rating, provider_order.get(c.provider.id, 99), c.model_id))
    return chosen


# --------------------------------------------------------------------------
# Wire translations. Every helper is a pure function of its arguments.
# The provider observes system text, the last user message's text, tool
# names, max_tokens, stop sequences, image bytes and JSON mode; each
# translation below preserves exactly those, in the target wire's shape.
# --------------------------------------------------------------------------

def blocks_text(content):
    """The text a provider's parser reads from a content value."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if is_record(block) and block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts)
    return ""


def openai_system_text(body):
    parts = []
    messages = body.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if is_record(message) and message.get("role") == "system":
                text = blocks_text(message.get("content"))
                if text != "":
                    parts.append(text)
    return "\n".join(parts)


def max_tokens_of_openai(body):
    value = body.get("max_tokens")
    if not isinstance(value, (int, float)):
        value = body.get("max_completion_tokens")
    return value if isinstance(value, (int, float)) else None


def stop_list(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [v for v in value if isinstance(v, str)]
    return []


def image_part_from_openai(block):
    """An OpenAI image_url part as (kind, data): base64 payload or URL."""
    image_url = block.get("image_url")
    if not is_record(image_url):
        return None
    url = image_url.get("url")
    if not isinstance(url, str) or not url:
        return None
    if url.startswith("data:"):
        header, _, payload = url.partition(",")
        media = header[len("data:"):].split(";")[0] or "image/png"
        return ("base64", media, payload)
    return ("url", None, url)


def image_part_from_anthropic(block):
    source = block.get("source")
    if not is_record(source):
        return None
    if source.get("type") == "base64" and isinstance(source.get("data"), str):
        media = source.get("media_type") if isinstance(source.get("media_type"), str) else "image/png"
        return ("base64", media, source["data"])
    if source.get("type") == "url" and isinstance(source.get("url"), str):
        return ("url", None, source["url"])
    return None


# --- client OpenAI -> upstream wires -------------------------------------

def openai_passthrough(body, model_id, stream):
    out = dict(body)
    out["model"] = model_id
    if stream:
        out["stream"] = True
        options = dict(body.get("stream_options")) if is_record(body.get("stream_options")) else {}
        options["include_usage"] = True
        out["stream_options"] = options
    return out


def openai_to_anthropic(body, model_id, stream):
    out = {"model": model_id}
    system = openai_system_text(body)
    if system != "":
        out["system"] = system
    messages = []
    for message in body.get("messages") or []:
        if not is_record(message) or message.get("role") == "system":
            continue
        role = message.get("role")
        content = message.get("content")
        if role == "tool":
            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": message.get("tool_call_id") if isinstance(message.get("tool_call_id"), str) else "",
                    "content": content if isinstance(content, str) else blocks_text(content),
                }],
            })
            continue
        if role == "assistant":
            blocks = []
            text = blocks_text(content)
            if text != "":
                blocks.append({"type": "text", "text": text})
            calls = message.get("tool_calls")
            if isinstance(calls, list):
                for call in calls:
                    if not is_record(call) or not is_record(call.get("function")):
                        continue
                    fn = call["function"]
                    args = fn.get("arguments")
                    parsed_args = {}
                    if isinstance(args, str) and args.strip():
                        try:
                            value = json.loads(args)
                            if is_record(value):
                                parsed_args = value
                        except ValueError:
                            parsed_args = {}
                    elif is_record(args):
                        parsed_args = args
                    blocks.append({
                        "type": "tool_use",
                        "id": call.get("id") if isinstance(call.get("id"), str) else "",
                        "name": fn.get("name") if isinstance(fn.get("name"), str) else "",
                        "input": parsed_args,
                    })
            if blocks:
                messages.append({"role": "assistant", "content": blocks})
            continue
        # user (or anything else) becomes a user message
        if isinstance(content, str):
            messages.append({"role": "user", "content": content})
            continue
        blocks = []
        if isinstance(content, list):
            for block in content:
                if not is_record(block):
                    continue
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    blocks.append({"type": "text", "text": block["text"]})
                elif block.get("type") == "image_url":
                    image = image_part_from_openai(block)
                    if image is None:
                        continue
                    kind, media, payload = image
                    if kind == "base64":
                        blocks.append({"type": "image", "source": {"type": "base64", "media_type": media, "data": payload}})
                    else:
                        blocks.append({"type": "image", "source": {"type": "url", "url": payload}})
        if blocks:
            messages.append({"role": "user", "content": blocks})
    out["messages"] = messages
    cap = max_tokens_of_openai(body)
    if cap is not None:
        out["max_tokens"] = cap
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        out["top_p"] = body["top_p"]
    stops = stop_list(body.get("stop"))
    if stops:
        out["stop_sequences"] = stops
    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        converted = []
        for tool in tools:
            if is_record(tool) and is_record(tool.get("function")) and isinstance(tool["function"].get("name"), str):
                fn = tool["function"]
                converted.append({
                    "name": fn["name"],
                    "description": fn.get("description") if isinstance(fn.get("description"), str) else "",
                    "input_schema": fn.get("parameters") if is_record(fn.get("parameters")) else {"type": "object", "properties": {}},
                })
        if converted:
            out["tools"] = converted
    choice = body.get("tool_choice")
    if choice == "auto":
        out["tool_choice"] = {"type": "auto"}
    elif choice == "required":
        out["tool_choice"] = {"type": "any"}
    elif choice == "none":
        out["tool_choice"] = {"type": "none"}
    elif is_record(choice) and is_record(choice.get("function")) and isinstance(choice["function"].get("name"), str):
        out["tool_choice"] = {"type": "tool", "name": choice["function"]["name"]}
    if stream:
        out["stream"] = True
    return out


def gemini_contents_from_openai(body):
    contents = []
    tool_names = {}
    for message in body.get("messages") or []:
        if not is_record(message) or message.get("role") == "system":
            continue
        role = message.get("role")
        content = message.get("content")
        if role == "tool":
            call_id = message.get("tool_call_id")
            name = tool_names.get(call_id, "tool")
            text = content if isinstance(content, str) else blocks_text(content)
            result = text
            try:
                parsed = json.loads(text)
                if is_record(parsed):
                    result = parsed
            except (ValueError, TypeError):
                result = {"result": text}
            if not is_record(result):
                result = {"result": result}
            contents.append({"role": "user", "parts": [{"functionResponse": {"name": name, "response": result}}]})
            continue
        parts = []
        if isinstance(content, str):
            if content != "":
                parts.append({"text": content})
        elif isinstance(content, list):
            for block in content:
                if not is_record(block):
                    continue
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append({"text": block["text"]})
                elif block.get("type") == "image_url":
                    image = image_part_from_openai(block)
                    if image is None:
                        continue
                    kind, media, payload = image
                    if kind == "base64":
                        parts.append({"inlineData": {"mimeType": media, "data": payload}})
                    else:
                        parts.append({"fileData": {"fileUri": payload}})
        if role == "assistant":
            calls = message.get("tool_calls")
            if isinstance(calls, list):
                for call in calls:
                    if not is_record(call) or not is_record(call.get("function")):
                        continue
                    fn = call["function"]
                    name = fn.get("name") if isinstance(fn.get("name"), str) else ""
                    if isinstance(call.get("id"), str):
                        tool_names[call["id"]] = name
                    args = fn.get("arguments")
                    parsed_args = {}
                    if isinstance(args, str) and args.strip():
                        try:
                            value = json.loads(args)
                            if is_record(value):
                                parsed_args = value
                        except ValueError:
                            parsed_args = {}
                    elif is_record(args):
                        parsed_args = args
                    parts.append({"functionCall": {"name": name, "args": parsed_args}})
            contents.append({"role": "model", "parts": parts})
        else:
            contents.append({"role": "user", "parts": parts})
    return contents


def gemini_tools_from_openai(body):
    tools = body.get("tools")
    if not isinstance(tools, list) or not tools:
        return None, None
    declarations = []
    for tool in tools:
        if is_record(tool) and is_record(tool.get("function")) and isinstance(tool["function"].get("name"), str):
            fn = tool["function"]
            declarations.append({
                "name": fn["name"],
                "description": fn.get("description") if isinstance(fn.get("description"), str) else "",
                "parameters": fn.get("parameters") if is_record(fn.get("parameters")) else {"type": "object", "properties": {}},
            })
    if not declarations:
        return None, None
    config = None
    choice = body.get("tool_choice")
    if choice == "required":
        config = {"functionCallingConfig": {"mode": "ANY"}}
    elif choice == "none":
        config = {"functionCallingConfig": {"mode": "NONE"}}
    elif choice == "auto":
        config = {"functionCallingConfig": {"mode": "AUTO"}}
    elif is_record(choice) and is_record(choice.get("function")) and isinstance(choice["function"].get("name"), str):
        config = {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": [choice["function"]["name"]]}}
    return [{"functionDeclarations": declarations}], config


def openai_to_gemini(body, model_id, stream, json_mode):
    out = {}
    system = openai_system_text(body)
    if system != "":
        out["systemInstruction"] = {"parts": [{"text": system}]}
    out["contents"] = gemini_contents_from_openai(body)
    tools, tool_config = gemini_tools_from_openai(body)
    if tools is not None:
        out["tools"] = tools
    if tool_config is not None:
        out["toolConfig"] = tool_config
    config = {}
    cap = max_tokens_of_openai(body)
    if cap is not None:
        config["maxOutputTokens"] = cap
    if isinstance(body.get("temperature"), (int, float)):
        config["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        config["topP"] = body["top_p"]
    stops = stop_list(body.get("stop"))
    if stops:
        config["stopSequences"] = stops
    if json_mode:
        config["responseMimeType"] = "application/json"
    if config:
        out["generationConfig"] = config
    return out


# --- client Anthropic -> upstream wires ----------------------------------

def anthropic_passthrough(body, model_id, stream):
    out = dict(body)
    out["model"] = model_id
    if stream:
        out["stream"] = True
    return out


def anthropic_to_openai(body, model_id, stream):
    out = {"model": model_id}
    messages = []
    system = blocks_text(body.get("system"))
    if system != "":
        messages.append({"role": "system", "content": system})
    for message in body.get("messages") or []:
        if not is_record(message):
            continue
        role = message.get("role")
        content = message.get("content")
        if isinstance(content, str):
            messages.append({"role": "assistant" if role == "assistant" else "user", "content": content})
            continue
        if not isinstance(content, list):
            continue
        if role == "assistant":
            text = blocks_text(content)
            calls = []
            for block in content:
                if is_record(block) and block.get("type") == "tool_use":
                    calls.append({
                        "id": block.get("id") if isinstance(block.get("id"), str) else "",
                        "type": "function",
                        "function": {
                            "name": block.get("name") if isinstance(block.get("name"), str) else "",
                            "arguments": json.dumps(block.get("input") if is_record(block.get("input")) else {}),
                        },
                    })
            entry = {"role": "assistant", "content": text if text != "" else None}
            if calls:
                entry["tool_calls"] = calls
            messages.append(entry)
            continue
        parts = []
        for block in content:
            if not is_record(block):
                continue
            if block.get("type") == "tool_result":
                result_content = block.get("content")
                if isinstance(result_content, str):
                    text = result_content
                elif isinstance(result_content, list):
                    text = blocks_text(result_content)
                    if text == "":
                        text = json.dumps(result_content)
                elif result_content is None:
                    text = ""
                else:
                    text = json.dumps(result_content)
                messages.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id") if isinstance(block.get("tool_use_id"), str) else "",
                    "content": text,
                })
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append({"type": "text", "text": block["text"]})
            elif block.get("type") == "image":
                image = image_part_from_anthropic(block)
                if image is None:
                    continue
                kind, media, payload = image
                if kind == "base64":
                    parts.append({"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (media, payload)}})
                else:
                    parts.append({"type": "image_url", "image_url": {"url": payload}})
        if parts:
            messages.append({"role": "user", "content": parts})
    out["messages"] = messages
    if isinstance(body.get("max_tokens"), (int, float)):
        out["max_tokens"] = body["max_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        out["top_p"] = body["top_p"]
    stops = stop_list(body.get("stop_sequences"))
    if stops:
        out["stop"] = stops
    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        converted = []
        for tool in tools:
            if is_record(tool) and isinstance(tool.get("name"), str):
                converted.append({
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description") if isinstance(tool.get("description"), str) else "",
                        "parameters": tool.get("input_schema") if is_record(tool.get("input_schema")) else {"type": "object", "properties": {}},
                    },
                })
        if converted:
            out["tools"] = converted
    choice = body.get("tool_choice")
    if is_record(choice):
        if choice.get("type") == "auto":
            out["tool_choice"] = "auto"
        elif choice.get("type") == "any":
            out["tool_choice"] = "required"
        elif choice.get("type") == "none":
            out["tool_choice"] = "none"
        elif choice.get("type") == "tool" and isinstance(choice.get("name"), str):
            out["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
    if stream:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


def anthropic_to_gemini(body, model_id, stream):
    out = {}
    system = blocks_text(body.get("system"))
    if system != "":
        out["systemInstruction"] = {"parts": [{"text": system}]}
    contents = []
    tool_names = {}
    for message in body.get("messages") or []:
        if not is_record(message):
            continue
        role = message.get("role")
        content = message.get("content")
        if isinstance(content, str):
            contents.append({"role": "model" if role == "assistant" else "user",
                             "parts": [{"text": content}] if content != "" else []})
            continue
        if not isinstance(content, list):
            continue
        parts = []
        for block in content:
            if not is_record(block):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append({"text": block["text"]})
            elif block.get("type") == "image":
                image = image_part_from_anthropic(block)
                if image is None:
                    continue
                kind, media, payload = image
                if kind == "base64":
                    parts.append({"inlineData": {"mimeType": media, "data": payload}})
                else:
                    parts.append({"fileData": {"fileUri": payload}})
            elif block.get("type") == "tool_use" and role == "assistant":
                name = block.get("name") if isinstance(block.get("name"), str) else ""
                if isinstance(block.get("id"), str):
                    tool_names[block["id"]] = name
                parts.append({"functionCall": {"name": name, "args": block.get("input") if is_record(block.get("input")) else {}}})
            elif block.get("type") == "tool_result":
                call_id = block.get("tool_use_id")
                name = tool_names.get(call_id, "tool")
                result_content = block.get("content")
                if isinstance(result_content, str):
                    try:
                        parsed = json.loads(result_content)
                        result = parsed if is_record(parsed) else {"result": parsed}
                    except ValueError:
                        result = {"result": result_content}
                elif isinstance(result_content, list):
                    result = {"result": blocks_text(result_content)}
                elif result_content is None:
                    result = {"result": ""}
                else:
                    result = {"result": result_content}
                parts.append({"functionResponse": {"name": name, "response": result}})
        contents.append({"role": "model" if role == "assistant" else "user", "parts": parts})
    out["contents"] = contents
    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        declarations = []
        for tool in tools:
            if is_record(tool) and isinstance(tool.get("name"), str):
                declarations.append({
                    "name": tool["name"],
                    "description": tool.get("description") if isinstance(tool.get("description"), str) else "",
                    "parameters": tool.get("input_schema") if is_record(tool.get("input_schema")) else {"type": "object", "properties": {}},
                })
        if declarations:
            out["tools"] = [{"functionDeclarations": declarations}]
            choice = body.get("tool_choice")
            if is_record(choice):
                if choice.get("type") == "any":
                    out["toolConfig"] = {"functionCallingConfig": {"mode": "ANY"}}
                elif choice.get("type") == "none":
                    out["toolConfig"] = {"functionCallingConfig": {"mode": "NONE"}}
                elif choice.get("type") == "auto":
                    out["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}
                elif choice.get("type") == "tool" and isinstance(choice.get("name"), str):
                    out["toolConfig"] = {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": [choice["name"]]}}
    config = {}
    if isinstance(body.get("max_tokens"), (int, float)):
        config["maxOutputTokens"] = body["max_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        config["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        config["topP"] = body["top_p"]
    stops = stop_list(body.get("stop_sequences"))
    if stops:
        config["stopSequences"] = stops
    if config:
        out["generationConfig"] = config
    return out


def build_upstream_body(client_api, client_body, candidate, stream):
    wire = candidate.provider.wire
    if client_api == "openai":
        if wire == "openai":
            return openai_passthrough(client_body, candidate.model_id, stream)
        if wire == "anthropic":
            return openai_to_anthropic(client_body, candidate.model_id, stream)
        fmt = client_body.get("response_format")
        json_mode = is_record(fmt) and fmt.get("type") in ("json_object", "json_schema")
        return openai_to_gemini(client_body, candidate.model_id, stream, json_mode)
    # anthropic client
    if wire == "anthropic":
        return anthropic_passthrough(client_body, candidate.model_id, stream)
    if wire == "openai":
        return anthropic_to_openai(client_body, candidate.model_id, stream)
    return anthropic_to_gemini(client_body, candidate.model_id, stream)


# --------------------------------------------------------------------------
# Upstream answers: extract (text, input tokens, output tokens, stop).
# --------------------------------------------------------------------------

STOP_FAMILY = {
    "stop": "end", "end_turn": "end", "STOP": "end",
    "length": "length", "max_tokens": "length", "MAX_TOKENS": "length",
    "tool_calls": "tool", "function_call": "tool", "tool_use": "tool",
    "content_filter": "filter", "refusal": "filter", "SAFETY": "filter",
}


def stop_for_client(family, client_api):
    if client_api == "openai":
        return {"end": "stop", "length": "length", "tool": "tool_calls", "filter": "content_filter"}.get(family, "stop")
    return {"end": "end_turn", "length": "max_tokens", "tool": "tool_use", "filter": "refusal"}.get(family, "end_turn")


def sse_payloads(data):
    text = data.decode("utf-8", "replace").replace("\r\n", "\n")
    payloads = []
    for frame in text.split("\n\n"):
        for line in frame.split("\n"):
            if line.startswith("data:"):
                payloads.append(line[len("data:"):].strip())
    return payloads


def sse_events(data):
    text = data.decode("utf-8", "replace").replace("\r\n", "\n")
    events = []
    for frame in text.split("\n\n"):
        if not frame.strip():
            continue
        event = None
        payload = None
        for line in frame.split("\n"):
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                payload = line[len("data:"):].strip()
        if payload is not None:
            events.append((event, payload))
    return events


def extract_answer(wire, data, stream):
    """(text, in_tokens, out_tokens, stop_family) or None when unusable."""
    try:
        if wire == "openai":
            if stream:
                text_parts = []
                usage_in = usage_out = 0
                finish = None
                for payload in sse_payloads(data):
                    if payload == "[DONE]":
                        continue
                    chunk = json.loads(payload)
                    if is_record(chunk.get("usage")):
                        usage_in = chunk["usage"].get("prompt_tokens") or 0
                        usage_out = chunk["usage"].get("completion_tokens") or 0
                    choices = chunk.get("choices") or []
                    if choices and is_record(choices[0]):
                        delta = choices[0].get("delta") or {}
                        if isinstance(delta.get("content"), str):
                            text_parts.append(delta["content"])
                        if isinstance(choices[0].get("finish_reason"), str):
                            finish = choices[0]["finish_reason"]
                return "".join(text_parts), usage_in, usage_out, STOP_FAMILY.get(finish, "end")
            payload = json.loads(data.decode("utf-8"))
            choices = payload.get("choices") or []
            if not choices:
                return None
            message = choices[0].get("message") or {}
            content = message.get("content")
            text = content if isinstance(content, str) else blocks_text(content)
            usage = payload.get("usage") or {}
            return (text, usage.get("prompt_tokens") or 0, usage.get("completion_tokens") or 0,
                    STOP_FAMILY.get(choices[0].get("finish_reason"), "end"))
        if wire == "anthropic":
            if stream:
                text_parts = []
                usage_in = usage_out = 0
                stop = None
                for event, payload in sse_events(data):
                    chunk = json.loads(payload)
                    if event == "message_start":
                        usage = (chunk.get("message") or {}).get("usage") or {}
                        usage_in = usage.get("input_tokens") or 0
                    elif event == "content_block_delta":
                        delta = chunk.get("delta") or {}
                        if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                            text_parts.append(delta["text"])
                    elif event == "message_delta":
                        usage = chunk.get("usage") or {}
                        usage_out = usage.get("output_tokens") or 0
                        delta = chunk.get("delta") or {}
                        if isinstance(delta.get("stop_reason"), str):
                            stop = delta["stop_reason"]
                return "".join(text_parts), usage_in, usage_out, STOP_FAMILY.get(stop, "end")
            payload = json.loads(data.decode("utf-8"))
            text = ""
            content = payload.get("content")
            if isinstance(content, list):
                text = "".join(b.get("text") for b in content
                               if is_record(b) and b.get("type") == "text" and isinstance(b.get("text"), str))
            usage = payload.get("usage") or {}
            return (text, usage.get("input_tokens") or 0, usage.get("output_tokens") or 0,
                    STOP_FAMILY.get(payload.get("stop_reason"), "end"))
        # gemini
        if stream:
            text_parts = []
            usage_in = usage_out = 0
            finish = None
            for payload in sse_payloads(data):
                chunk = json.loads(payload)
                candidates = chunk.get("candidates") or []
                if candidates and is_record(candidates[0]):
                    parts = (candidates[0].get("content") or {}).get("parts") or []
                    for part in parts:
                        if is_record(part) and isinstance(part.get("text"), str):
                            text_parts.append(part["text"])
                    if isinstance(candidates[0].get("finishReason"), str):
                        finish = candidates[0]["finishReason"]
                metadata = chunk.get("usageMetadata")
                if is_record(metadata):
                    usage_in = metadata.get("promptTokenCount") or 0
                    usage_out = metadata.get("candidatesTokenCount") or 0
            return "".join(text_parts), usage_in, usage_out, STOP_FAMILY.get(finish, "end")
        payload = json.loads(data.decode("utf-8"))
        candidates = payload.get("candidates") or []
        if not candidates:
            return None
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text") for p in parts if is_record(p) and isinstance(p.get("text"), str))
        metadata = payload.get("usageMetadata") or {}
        return (text, metadata.get("promptTokenCount") or 0, metadata.get("candidatesTokenCount") or 0,
                STOP_FAMILY.get(candidates[0].get("finishReason"), "end"))
    except (ValueError, UnicodeDecodeError, AttributeError, TypeError, IndexError):
        return None


# --------------------------------------------------------------------------
# Client answers.
# --------------------------------------------------------------------------

def openai_completion(model_id, text, usage_in, usage_out, family):
    return {
        "id": "chatcmpl-agg",
        "object": "chat.completion",
        "created": FIXED_CREATED,
        "model": model_id,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": stop_for_client(family, "openai"),
            "logprobs": None,
        }],
        "usage": {
            "prompt_tokens": usage_in,
            "completion_tokens": usage_out,
            "total_tokens": usage_in + usage_out,
        },
    }


def anthropic_message(model_id, text, usage_in, usage_out, family):
    return {
        "id": "msg_agg",
        "type": "message",
        "role": "assistant",
        "model": model_id,
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_for_client(family, "anthropic"),
        "stop_sequence": None,
        "usage": {"input_tokens": usage_in, "output_tokens": usage_out},
    }


def sse_frame(event, payload):
    if event is None:
        return "data: %s\n\n" % json.dumps(payload)
    return "event: %s\ndata: %s\n\n" % (event, json.dumps(payload))


def openai_stream_body(model_id, text, usage_in, usage_out, family):
    chunk_base = {"id": "chatcmpl-agg", "object": "chat.completion.chunk", "created": FIXED_CREATED, "model": model_id}
    frames = []
    first = dict(chunk_base)
    first["choices"] = [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]
    frames.append(sse_frame(None, first))
    if text:
        body = dict(chunk_base)
        body["choices"] = [{"index": 0, "delta": {"content": text}, "finish_reason": None}]
        frames.append(sse_frame(None, body))
    finish = dict(chunk_base)
    finish["choices"] = [{"index": 0, "delta": {}, "finish_reason": stop_for_client(family, "openai")}]
    frames.append(sse_frame(None, finish))
    usage_chunk = dict(chunk_base)
    usage_chunk["choices"] = []
    usage_chunk["usage"] = {"prompt_tokens": usage_in, "completion_tokens": usage_out, "total_tokens": usage_in + usage_out}
    frames.append(sse_frame(None, usage_chunk))
    frames.append("data: [DONE]\n\n")
    return "".join(frames).encode("utf-8")


def anthropic_stream_body(model_id, text, usage_in, usage_out, family):
    frames = []
    frames.append(sse_frame("message_start", {
        "type": "message_start",
        "message": {
            "id": "msg_agg", "type": "message", "role": "assistant", "model": model_id,
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": usage_in, "output_tokens": 0},
        },
    }))
    frames.append(sse_frame("content_block_start", {
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "text", "text": ""},
    }))
    if text:
        frames.append(sse_frame("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": text},
        }))
    frames.append(sse_frame("content_block_stop", {"type": "content_block_stop", "index": 0}))
    frames.append(sse_frame("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_for_client(family, "anthropic"), "stop_sequence": None},
        "usage": {"output_tokens": usage_out},
    }))
    frames.append(sse_frame("message_stop", {"type": "message_stop"}))
    return "".join(frames).encode("utf-8")


def openai_error(status, message, err_type="api_error", code=None):
    return status, {"error": {"message": message, "type": err_type, "code": code}}


def anthropic_error(status, message, err_type="api_error"):
    return status, {"type": "error", "error": {"type": err_type, "message": message}}


# --------------------------------------------------------------------------
# Upstream call + failover.
# --------------------------------------------------------------------------

def call_provider(candidate, client_api, client_body, stream):
    provider = candidate.provider
    base = provider.dial_base()
    upstream_body = build_upstream_body(client_api, client_body, candidate, stream)
    headers = dict(auth_headers(provider))
    if provider.wire == "gemini":
        action = "streamGenerateContent" if stream else "generateContent"
        url = "%s/v1beta/models/%s:%s" % (base, candidate.model_id, action)
        if stream:
            url += "?alt=sse"
    elif provider.wire == "anthropic":
        url = base + "/v1/messages"
        headers["anthropic-version"] = ANTHROPIC_VERSION
    else:
        url = base + "/v1/chat/completions"
    if stream:
        headers["accept"] = "text/event-stream"
    status, _resp_headers, data = http_request(
        "POST", url, body=json.dumps(upstream_body), headers=headers, timeout=UPSTREAM_TIMEOUT)
    return status, data


def serve_request(client_api, client_body, requires):
    """Try candidates in rating order; return (status, content_type, bytes)."""
    stream = client_body.get("stream") is True
    candidates = rank_candidates(requires, client_api, client_body)
    for candidate in candidates:
        status, data = call_provider(candidate, client_api, client_body, stream)
        if status != 200:
            continue
        extracted = extract_answer(candidate.provider.wire, data, stream)
        if extracted is None:
            continue
        text, usage_in, usage_out, family = extracted
        if client_api == "openai":
            if stream:
                return 200, "text/event-stream", openai_stream_body(candidate.model_id, text, usage_in, usage_out, family)
            return 200, "application/json", json.dumps(
                openai_completion(candidate.model_id, text, usage_in, usage_out, family)).encode("utf-8")
        if stream:
            return 200, "text/event-stream", anthropic_stream_body(candidate.model_id, text, usage_in, usage_out, family)
        return 200, "application/json", json.dumps(
            anthropic_message(candidate.model_id, text, usage_in, usage_out, family)).encode("utf-8")
    if client_api == "openai":
        status, payload = openai_error(503, "no free model could serve this request", "api_error", "no_free_model")
    else:
        status, payload = anthropic_error(503, "no free model could serve this request", "overloaded_error")
    return status, "application/json", json.dumps(payload).encode("utf-8")


# --------------------------------------------------------------------------
# The HTTP server facing the env's client.
# --------------------------------------------------------------------------

MANIFEST_INDEX = [0]


def next_requirements(client_api, body):
    index = MANIFEST_INDEX[0]
    MANIFEST_INDEX[0] = index + 1
    if index < len(BRIEF.manifest):
        return dict(BRIEF.manifest[index].get("requires") or {})
    return body_requirements(body, client_api)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send(self, status, content_type, data):
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        if data:
            self.wfile.write(data)

    def _send_json(self, status, payload):
        self._send(status, "application/json", json.dumps(payload).encode("utf-8"))

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/v1/models":
            ids = served_model_ids()
            display = {}
            for candidate in CANDIDATES:
                if candidate.free and candidate.model_id not in display:
                    display[candidate.model_id] = candidate.display_name
            if self.headers.get("x-api-key") is not None:
                self._send_json(200, {
                    "data": [{"type": "model", "id": mid, "display_name": display.get(mid, mid),
                              "created_at": FIXED_CREATED_AT} for mid in ids],
                    "has_more": False,
                })
            else:
                owners = {}
                for candidate in CANDIDATES:
                    if candidate.free and candidate.model_id not in owners:
                        owners[candidate.model_id] = candidate.provider.id
                self._send_json(200, {
                    "object": "list",
                    "data": [{"id": mid, "object": "model", "created": FIXED_CREATED,
                              "owned_by": owners.get(mid, "aggregator")} for mid in ids],
                })
            return
        self._send_json(404, {"error": {"message": "no route %s" % path, "type": "not_found", "code": None}})

    def do_POST(self):
        path = urlsplit(self.path).path
        if path not in ("/v1/chat/completions", "/v1/messages"):
            self._send_json(404, {"error": {"message": "no route %s" % path, "type": "not_found", "code": None}})
            return
        client_api = "openai" if path == "/v1/chat/completions" else "anthropic"
        try:
            length = int(self.headers.get("content-length") or 0)
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            body = json.loads(raw.decode("utf-8")) if raw else None
        except (ValueError, UnicodeDecodeError):
            body = None
        if not is_record(body):
            if client_api == "openai":
                status, payload = openai_error(400, "the request body is not one JSON object", "invalid_request_error")
            else:
                status, payload = anthropic_error(400, "the request body is not one JSON object", "invalid_request_error")
            self._send_json(status, payload)
            return
        requires = next_requirements(client_api, body)
        status, content_type, data = serve_request(client_api, body, requires)
        self._send(status, content_type, data)


# --------------------------------------------------------------------------
# Register, poll, main.
# --------------------------------------------------------------------------

def register(port):
    body = json.dumps({"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": CATALOG})
    status, _headers, data = http_request(
        "POST", ENV_URL + "/register", body=body, headers={"content-type": "application/json"}, timeout=PROBE_TIMEOUT)
    if status != 200:
        raise RuntimeError("POST /register answered HTTP %s: %s" % (status, data[:200]))
    reply = json.loads(data.decode("utf-8"))
    log("registered: the workload holds %s requests" % reply.get("requests"))


def poll_until_done(server):
    while True:
        status, _headers, data = http_request("GET", ENV_URL + "/work", timeout=2.0)
        if status == 200:
            try:
                work = json.loads(data.decode("utf-8"))
            except ValueError:
                work = {}
            if work.get("done") is True:
                return
        time.sleep(POLL_INTERVAL)


def main(argv):
    build_catalog()
    if "--list" in argv:
        sys.stdout.write(json.dumps(CATALOG) + "\n")
        sys.stdout.flush()
        return 0
    if not ENV_URL:
        log("MAKEFUN_ENV_URL is not set")
        return 1
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    port = server.server_address[1]
    if port == 8080:
        log("the OS picked 8080, which the env owns on one layout")
        return 1
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    log("listening on 0.0.0.0:%d with %d providers and %d free models"
        % (port, len(CATALOG["providers"]), len(served_model_ids())))
    try:
        register(port)
    except Exception as err:
        log("register failed: %s" % err)
        server.shutdown()
        return 1
    try:
        poll_until_done(server)
    finally:
        server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
