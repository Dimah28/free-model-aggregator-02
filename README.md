# free-model-aggregator-02

One running server that aggregates OpenRouter's free models behind a single
endpoint: an OpenAI-compatible Chat Completions API and an
Anthropic-compatible Messages API, in JSON and in server-sent events.

Every answer comes from a real free model dialled live through the
grader's gateway. The server picks the model per prompt: free models that
meet the prompt's requirements (tools, vision, JSON mode, minimum context),
ordered by the house quality rating in the brief — highest first, id
ascending as tie-break. When a provider refuses (429 / 5xx), the server
fails over to the next candidate and reports the refusals on the
`x-makefun-upstream-failures` header.

## Run

```sh
python3 main.py            # listen, register with the env, serve, exit on done
python3 main.py --list     # print the catalog document and exit 0
```

Python 3.13, standard library only, one file (`main.py`) in the repo root.
The brief comes from `$MAKEFUN_PROMPT_FILE`; the env from
`$MAKEFUN_ENV_URL`; the gateway from `$OPENAI_BASE_URL` with the trial key
in `$OPENAI_API_KEY`.

## Endpoints

- `POST /v1/chat/completions` — OpenAI Chat Completions, JSON or SSE
- `POST /v1/messages` — Anthropic Messages, JSON or SSE (translated to
  Chat Completions for the gateway, translated back for the client)
- `GET /v1/models` — OpenAI list shape for `Authorization: Bearer`,
  Anthropic list shape for `x-api-key`

## Catalog

14 free OpenRouter models (`openrouter/<vendor>/<model>:free`), read from
the brief at runtime — never hardcoded. `--list` prints exactly the
document the server registers with the env.

## Structure

- `main.py` — the whole server: brief parsing, catalog, rating-ordered
  routing with trial-local refusal memory, gateway dialling, both wires
  (JSON + streaming), failover header, env register/poll loop
- `LICENSE` — MIT
