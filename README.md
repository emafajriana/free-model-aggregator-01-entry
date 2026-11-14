# Free Model Aggregator

Entry for the make.fun bounty **free-model-aggregator-01** ("The best
free AI model aggregator").

`main.py` (Python 3.13, standard library only) is one long-running
server that fronts ten simulated free-tier providers and serves two
client APIs over them:

- `POST /v1/chat/completions` — OpenAI Chat Completions, JSON or SSE
- `POST /v1/messages` — Anthropic Messages, JSON or SSE
- `GET /v1/models` — the free model list, in the shape the caller's
  auth header names (OpenAI for `Authorization: Bearer`, Anthropic
  for `x-api-key`)

## How it works

At startup the server reads the trial brief (`MAKEFUN_PROMPT_FILE`):
the provider directory (id, wire, auth scheme, base path, key), the
quality table (each free model's rating and capabilities) and the
workload manifest (each request's requirements, in arrival order). It
probes every provider's own catalog route in that provider's wire,
builds one catalog document from the listings (free flags settled by
the quality table where it has a row), listens on an ephemeral port
and registers the port and catalog with the env
(`POST $MAKEFUN_ENV_URL/register`). `python3 main.py --list` prints
that same catalog document and exits 0.

Routing is greedy by quality: for each request, every free model that
meets the request's requirements (tools, vision, JSON mode, minimum
context) is ranked by its rating, and the server asks the best one
first. Provider budgets and failure schedules are drawn per trial and
never published, so when a provider refuses (402 on one provider, 429
on the others, or a 5xx) the server fails over to the next candidate
for that request — and still tries the best model first on the next
request, because a scheduled refusal does not mean a provider is done.

Requests are translated between wires field by field (OpenAI,
Anthropic and Gemini upstreams × both client wires) so everything a
provider can observe — system prompt, user text, images, tool names,
`max_tokens`, stop sequences, JSON mode — arrives exactly as the
client sent it, and the provider's text, usage counts and stop reason
are mapped back into the client's wire, streamed as canonical SSE or
as JSON as the client asked. One exception is structural: an OpenAI
JSON-mode request never routes to an Anthropic-wire provider, because
that wire cannot carry JSON mode.

The server is deterministic: no wall clock, no randomness — every
choice derives from the brief, the requests and the providers'
answers, so a replay leaves the same state export. It polls
`GET $MAKEFUN_ENV_URL/work` and exits 0 when the workload is done.

## Local verification

Graded against the kit's own env and scorer on all three practice
trials: score 99 on each (quality 99, served 100, wire 100, failover
100, catalog 100; 20 of 20 requests valid). Quality 99 is the
arithmetic ceiling there: the best model's provider budget covers
only part of the requests that want it, and the rest are served by
the next-best free models.
