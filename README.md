# ctxdb

**A context database for LLMs.** Store context as *structure*, retrieve only the
pieces that matter for the question at hand.

Not a text dump: ctxdb splits context into layers with different semantics and
different expiry rules, then retrieves across them with a search that combines
the lexical and the semantic. It runs on Python + SQLite (FTS5 + sqlite-vec) in
a single `.db` file, and plugs into Claude as an MCP server.

```
                  ┌──────────── your query ────────────┐
                  ▼                                    ▼
            BM25 / FTS5                          vector KNN
        (codes, names, acronyms)            (paraphrase, synonyms)
                  └──────────► RRF fusion ◄────────────┘
                                   │
                        rank by kind + confidence
                                   │
                         pack to token budget
                                   ▼
                        context ready for the model
```

---

## Quick start

```bash
git clone https://github.com/kguttas/ctxdb.git && cd ctxdb
uv venv --python 3.12
uv pip install -e .
```

Connect it to Claude Code:

```bash
claude mcp add ctxdb -- uv --directory /absolute/path/to/ctxdb run ctxdb-mcp
```

That's it. Claude now has nine tools; ask it to remember something and then ask
about it in a later session. No server to run, no Docker, no API key — the
default setup uses BM25 only, which needs zero extra dependencies and gets you
surprisingly far.

<details>
<summary>Claude Desktop instead (<code>claude_desktop_config.json</code>)</summary>

```json
{
  "mcpServers": {
    "ctxdb": {
      "command": "uv",
      "args": ["--directory", "/absolute/path/to/ctxdb", "run", "ctxdb-mcp"],
      "env": { "CTXDB_PATH": "/absolute/path/to/context.db" }
    }
  }
}
```
</details>

Prefer the terminal? Same engine, no Claude required:

```bash
ctxdb collection create project --embeddings none
ctxdb ingest project ./docs --pattern "*.md"
ctxdb search project "which search engine do we use"
```

---

## Why it is built this way

### The problem is not storing. It is separating.

A system that blindly cuts every 500 tokens splits tables in half, separates a
claim from the condition that qualifies it, and returns fragments that mean
nothing on their own. Here chunking follows the document's structure: it never
crosses a heading, never splits a fenced code block, and every chunk carries its
full heading path (`Manual > Billing > Credit notes`) — so it keeps its context
even when retrieved alone.

### Not all context ages the same way

| Layer | What it is | Expiry |
|---|---|---|
| `fact` | An atomic claim with a stable key | A new value **supersedes** the old one |
| `chunk` | A passage of a document, with its position | Lives as long as the document |
| `note` | Free-standing text, no source document | Permanent until deleted |

The `fact` / `chunk` split is what avoids the most expensive failure mode of
these systems: retrieving the old value *and* the new one, and letting the model
pick. Call `context_set_fact` with a key and the previous value is marked
superseded — it stops being retrieved, but stays in the database so you can
audit what was believed and since when.

```
context_set_fact("Rate limit is 100 rpm",  key="api.rate_limit")   → created
context_set_fact("Rate limit is 300 rpm",  key="api.rate_limit")   → superseded
context_search("what is the rate limit")   → only ever returns 300 rpm
```

### Hybrid search, because each branch sees what the other misses

Embeddings handle paraphrase (*"how do I void an invoice"* → a passage that only
ever says *"credit note"*). BM25 handles literals — a SKU, `HTTP 429`, a surname
— where vectors fail systematically. Both rankings are fused with **RRF**, which
sums `1/(k + rank)`: it works on *positions*, not scores, so nothing has to be
normalized between a cosine distance and a BM25 score, two scales that were
never comparable.

### A lightweight graph on top

`entities` + `relations` answer "what connects to what", which lets a query
recover material related to something without that material naming it. Entities
are declared explicitly on purpose — extracting them with heuristics fills the
graph with noise. Known entities are then auto-linked to new content by exact
name match, which keeps precision high.

### Packing to a token budget

The step almost nobody implements, and the one that pays off most. Returning 20
chunks "just in case" dilutes the model's attention. You ask for 1500 tokens and
get the best material that fits in 1500 tokens.

---

## Tools Claude gets

| Tool | When it fires |
|---|---|
| `context_search` | Before answering anything about stored material |
| `context_add_document` | Ingesting long text (manual, spec, transcript) |
| `context_set_fact` | A value that **changes**: decision, preference, state |
| `context_add_note` | A loose observation worth remembering |
| `context_recall_entity` | The question is about one specific thing |
| `context_relate` | Connecting two entities in the graph |
| `context_create_collection` | Isolating a new domain |
| `context_status` | Finding out what exists and where to search |
| `context_forget` | Permanent deletion by id |

`context_search` returns a labelled block, so the model can cite provenance and
you can tell when it answered with something that was never retrieved:

```
<context query="what is the rate limit" chunks="2">

[1] (fact) Payments API
The Payments API rate limit rose to 300 requests per minute in August 2026.

[2] (chunk) api.md
    section: Payments API > Limits
The limit is 100 requests per minute per token. Exceeding it returns HTTP 429.

</context>
```

---

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `CTXDB_PATH` | Path to the `.db` file | `~/.ctxdb/context.db` |
| `CTXDB_COLLECTION` | Default collection for tool calls | `default` |
| `CTXDB_EMBED` | Embedding spec for the default collection | `none` |
| `VOYAGE_API_KEY` | Only if a collection uses `voyage:...` | — |

### Choosing an embedding engine

Pinned **per collection** at creation time and stored on the collection row, so
one `.db` file can hold a local collection and an API-backed one side by side.
Changing it later requires a reindex, because vectors from different models are
not comparable — which is why they live in tables separated by dimension and
every item records the model that indexed it.

| Spec | When to use it | Install |
|---|---|---|
| `none` | Starting out. Zero dependencies; BM25 alone solves a lot | — |
| `local:intfloat/multilingual-e5-small` | Fully offline, 384 dims, strong multilingual | `uv pip install -e ".[local]"` |
| `voyage:voyage-3.5` | Best retrieval quality; text leaves your machine | `uv pip install -e ".[voyage]"` |

```bash
ctxdb collection create research --embeddings local
ctxdb collection create clients  --embeddings voyage     # same .db file
```

---

## Library usage

```python
from ctxdb import connect, get_or_create_collection, add_document, search, render_context

conn = connect("context.db")
get_or_create_collection(conn, "project", embed_spec="local:intfloat/multilingual-e5-small")
add_document(conn, "project", open("manual.md").read(), uri="manual.md", title="Manual")

result = search(conn, "project", "how do I void an invoice", budget_tokens=1200, neighbors=1)
print(render_context(result))
```

`search()` returns hits with `score`, `signals` (which branch found it),
`source` and `tokens`, so you can inspect *why* something was retrieved.

## CLI

```bash
ctxdb collection create project --embeddings local
ctxdb collection list
ctxdb ingest project ./docs --pattern "*.md"
ctxdb fact project "The store is SQLite with sqlite-vec" --key arch.db
ctxdb relate project "Batch" issued_by "Tax Authority"
ctxdb search project "which search engine do we use" --neighbors 1
ctxdb entity project "Tax Authority"
ctxdb status
```

## Tests

No network, no models, no fixtures to download:

```bash
python tests/test_ctxdb.py    # engine: chunking, supersession, budget, graph
python tests/test_vector.py   # vector plumbing, via a toy embedder
```

## Project layout

```
ctxdb/
  schema.py      SQL schema, and why each table exists
  db.py          Connection, sqlite-vec loading, per-dimension vector tables
  chunking.py    Heading-, code- and paragraph-aware splitting
  embeddings.py  Swappable providers (none / local / voyage)
  store.py       Writes: documents, facts, entities, relations
  retrieve.py    BM25 + vectors, RRF fusion, token-budget packing
  server.py      MCP server
  cli.py         Terminal interface
```

## Known limits

- Entities are not extracted automatically: they are declared, then auto-linked
  by exact name match. A deliberate trade of recall for precision.
- No cross-encoder reranker. RRF plus kind and confidence weighting holds up
  well into the tens of thousands of chunks.
- `context_search` does not rewrite the query. For very indirect questions, run
  two searches with different phrasings.
- Single-writer, like SQLite itself. Fine for one user and one agent; not built
  for a multi-tenant write load.

## License

MIT
