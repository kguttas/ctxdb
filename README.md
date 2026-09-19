# ctxdb

[![tests](https://github.com/kguttas/ctxdb/actions/workflows/ci.yml/badge.svg)](https://github.com/kguttas/ctxdb/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.12%20%7C%203.13-blue)](https://github.com/kguttas/ctxdb)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**A context database for LLMs.** Store context as *structure*, retrieve only the
pieces that matter for the question at hand.

Not a text dump: ctxdb splits context into layers with different semantics and
different expiry rules, then retrieves across them with a search that combines
the lexical and the semantic. It runs on Python + SQLite (FTS5 + sqlite-vec) in
a single `.db` file, and plugs into any MCP client — Claude Code, Claude Desktop,
Cocos — several of them sharing that one file at the same time.

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

Install it as a plugin, which brings the server, the recall hook and the
commands in one piece:

```
/plugin marketplace add kguttas/ctxdb
/plugin install ctxdb@ctxdb
/ctxdb:setup
```

`/ctxdb:setup` is a separate step on purpose: it downloads a few hundred
megabytes for local embeddings, and a session-start hook has no business doing
that behind your back. Pass `no-embeddings` to skip it and run on BM25 alone.

Restart afterwards — the MCP server and the hook are both read at launch.

<details>
<summary>Wiring it by hand instead</summary>

```bash
claude mcp add ctxdb -s user \
  -e CTXDB_CLIENT=claude \
  -e CTXDB_EMBED=local:intfloat/multilingual-e5-small \
  -- uv run --no-sync --project /absolute/path/to/ctxdb ctxdb-mcp
```

Two flags carry weight here. `--no-sync` is not an optimisation: a plain
`uv run` re-syncs the environment on every invocation, and on Windows that
collides with the DLLs the running server holds open — which corrupts the
environment it is trying to prepare. And `--project` rather than `--directory`,
because `--directory` changes the working directory, and the working directory
is what tells the server which project it is remembering for.

Doing it this way means also adding the hook and the policy yourself; see
[Making it actually fire](#making-it-actually-fire).
</details>

<details>
<summary>Claude Desktop instead (<code>claude_desktop_config.json</code>)</summary>

```json
{
  "mcpServers": {
    "ctxdb": {
      "command": "/absolute/path/to/ctxdb/.venv/bin/python",
      "args": ["-m", "ctxdb.server"],
      "env": {
        "CTXDB_PATH": "/absolute/path/to/context.db",
        "CTXDB_CLIENT": "claude-desktop",
        "CTXDB_COLLECTION": "desktop"
      }
    }
  }
}
```

Claude Desktop has no project directory to derive a collection from, so pin one
with `CTXDB_COLLECTION` rather than letting it land wherever the app happened to
be launched.
</details>

Prefer the terminal? Same engine, no Claude required:

```bash
ctxdb collection create project --embeddings none
ctxdb ingest project ./docs --pattern "*.md"
ctxdb search project "which search engine do we use"
```

---

## Making it actually fire

A retrieval system nobody queries stores perfectly and remembers nothing. This is
the failure mode worth naming, because it looks exactly like success: the server
connects, the tools appear in the list, and then months pass in which not one
thing is written or read. A tool description is not a trigger. It is read by a
model that is already thinking about storing something — which, unprompted, it
rarely is.

So the triggers live outside the model, in two pieces that do not depend on it
choosing well:

**Recall happens on its own.** A `SessionStart` hook injects what is already
stored for this project before the first question is asked:

```jsonc
// ~/.claude/settings.json
{
  "hooks": {
    "SessionStart": [
      { "hooks": [{ "type": "command", "timeout": 10,
                    "command": "/path/to/ctxdb/.venv/bin/python -m ctxdb.cli recall --hook" }] }
    ]
  }
}
```

`ctxdb recall --hook` prints live facts and notes for the current project plus the
global collection, capped at a token budget, wrapped in a block that labels itself
as recalled data rather than instruction. An empty store prints nothing at all, so
a project with no memory yet pays nothing for the hook. It takes about 200 ms.

**Writing is governed by a policy** — `ctxdb/policy.py`, injected by that same
hook — that says what clears the bar: a decision *and its reason*, a correction the user made, a
constraint that cost time to find. And what does not: anything git already
records, the narrative of what was just done, secrets, details that die with the
conversation. The bar matters more than the prose around it — a store full of
noise is how retrieval stops being worth reading.

A `PreCompact` hook adds a last reminder to persist what the conversation is
about to lose, and `/ctxdb:remember` and `/ctxdb:recall` give a manual override.

The policy travels in the package rather than in your `CLAUDE.md` because it has
to arrive with the plugin. Handing someone the tools and asking them to also
paste in a policy is asking them to do the one step whose omission is the entire
failure. If you keep your own copy in a `CLAUDE.md`, run the hook without
`--policy` so it is not stated twice.

Without this layer the rest of this README describes a very good database that
nothing writes to.

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

The lexical half is only as good as its tokenizer, and the settings that look like
housekeeping decide whether it works at all. FTS5 matches whole tokens, so without
a stemmer `deploy` never reaches *"deploys"* — the index uses `porter`, whose
first step also strips the Spanish plural. And a period is a token character
everywhere or nowhere: with `.` in `tokenchars`, *"returns HTTP 429."* indexed the
token `429.`, and a search for `429` found nothing. So did `ships` in *"…ships."*
Neither failure raises anything; both simply return no results, for years.

### Collections follow the project

The collection is derived from the working directory, with a `global` one searched
alongside it. A preference about how you like answers written belongs everywhere;
the reason this repository uses Postgres belongs in this repository. Both in one
collection means each project's memory is diluted by every other project's, and
`context_search` with no collection named searches the pair.

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
| `CTXDB_COLLECTION` | Pin the project collection instead of deriving it from the cwd | — |
| `CTXDB_GLOBAL` | Name of the cross-project collection | `global` |
| `CTXDB_EMBED` | Embedding spec for collections created from now on | `none` |
| `CTXDB_CLIENT` | Name recorded on everything this agent writes | `unknown` |
| `CTXDB_BUSY_TIMEOUT` | Milliseconds a writer waits for the lock | `20000` |
| `VOYAGE_API_KEY` | Only if a collection uses `voyage:...` | — |

### Several agents on one database

One `.db` file serves more than one coding agent at a time — Claude Code, Cocos,
whatever comes next. Point them all at the same `CTXDB_PATH` and give each one a
name:

```jsonc
// Claude Code
{ "command": "uv", "args": ["--directory", "/path/to/ctxdb", "run", "ctxdb-mcp"],
  "env": { "CTXDB_CLIENT": "claude" } }
```

```toml
# Cocos
[mcp.servers.ctxdb]
command = "uv"
args = ["--directory", "/path/to/ctxdb", "run", "ctxdb-mcp"]
env = { CTXDB_PATH = "~/.ctxdb/context.db", CTXDB_CLIENT = "cocos" }
```

What one stores, the others find. Every item records who wrote it, so a wrong
answer can be traced back to the session that planted it:

```sql
SELECT client, COUNT(*) FROM items GROUP BY client;
```

**Why it holds up.** SQLite admits one writer at a time, so what decides whether
this works is not the number of agents but how long each one holds the lock. WAL
lets readers carry on regardless; the writer's share is kept to the inserts alone,
with embeddings computed *before* the transaction opens. Ingesting a manual with a
slow model therefore blocks nobody — measured in `tests/test_concurrent.py`, where a
second agent writes straight through an eight-second ingest without losing a single
write. Before that split it lost all twelve.

If your machine is slow enough to still hit contention, raise `CTXDB_BUSY_TIMEOUT`:
waiting is the correct behaviour, since the alternative is not more concurrency but
a lost write.

### Choosing an embedding engine

Pinned **per collection** at creation time and stored on the collection row, so
one `.db` file can hold a local collection and an API-backed one side by side.
Vectors from different models are not comparable, which is why they live in tables
separated by dimension and every item records the model that indexed it — and why
switching engines means rebuilding them:

```bash
ctxdb collection reindex project --embeddings local
```

That re-embeds everything already stored, in batches, dropping the old vectors by
their old dimension first. It is how a collection that started on BM25 alone gains
semantic search over material ingested long before.

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
ctxdb collection reindex project --embeddings local   # switch engine, rebuild vectors
ctxdb ingest project ./docs --pattern "*.md"
ctxdb fact project "The store is SQLite with sqlite-vec" --key arch.db
ctxdb relate project "Batch" issued_by "Tax Authority"
ctxdb search project "which search engine do we use" --neighbors 1
ctxdb entity project "Tax Authority"
ctxdb recall                   # what a new session would be handed
ctxdb status
ctxdb serve                    # the MCP server over stdio, same as ctxdb-mcp
```

`--db /path/to/other.db` before the subcommand works on a file other than
`CTXDB_PATH`; `search` also takes `-k`, `--tokens`, `--kinds`, `--entities` and
`--json` for the raw hits instead of the rendered block. The CLI opens the same
file an agent may have open — it queues behind the writer like any other client.

## Tests

No network, no models, no fixtures to download:

```bash
python tests/test_ctxdb.py       # engine: chunking, supersession, budget, graph
python tests/test_vector.py      # vector plumbing, via a toy embedder
python tests/test_concurrent.py  # two agents writing at once, and the migration
```

CI runs the three suites on Linux and Windows against Python 3.10, 3.12 and 3.13.

## Project layout

```
ctxdb/
  schema.py      SQL schema, and why each table exists
  db.py          Connection, migrations, sqlite-vec, and the concurrency settings
  chunking.py    Heading-, code- and paragraph-aware splitting
  embeddings.py  Swappable providers (none / local / voyage)
  store.py       Writes: documents, facts, entities, relations
  retrieve.py    BM25 + vectors, RRF fusion, token-budget packing
  server.py      MCP server
  cli.py         Terminal interface
tests/
  test_ctxdb.py       Engine
  test_vector.py      Vector branch
  test_concurrent.py  Two processes on one file, and the migration
```

### Schema and upgrades

The schema is at **version 3**. Version 2 added the `client` column; version 3
rebuilt the lexical index with a stemming tokenizer. Upgrading is
just pulling the new code: `db._migrate` runs on every open, adds what is missing
and is a no-op once the file is current. Existing items simply carry a `NULL`
client, since nobody recorded one at the time.

The index rebuild is decided by comparing the tokenizer in the stored DDL against
the current declaration, not by a version number, which makes it self-correcting:
any file whose index does not match gets rebuilt, including one left behind by a
half-finished upgrade. It re-indexes from `items`, so nothing has to be
re-ingested — but note it only rebuilds the *lexical* index. Vectors are a
separate decision, and a deliberate one: `collection reindex`.

## Known limits

- Entities are not extracted automatically: they are declared, then auto-linked
  by exact name match. A deliberate trade of recall for precision.
- No cross-encoder reranker. RRF plus kind and confidence weighting holds up
  well into the tens of thousands of chunks.
- `context_search` does not rewrite the query. For very indirect questions, run
  two searches with different phrasings.
- Deciding *what* is worth storing is still the model's judgement. The hooks
  guarantee that memory is read and that the question gets asked at the right
  moment; nothing guarantees a good answer to it. Expect to prune.
- The collection is the working directory's name, so two checkouts of the same
  repository share a memory and two different projects with the same folder name
  collide. Pin `CTXDB_COLLECTION` per project where that matters.
- Single-writer, like SQLite itself. Several agents share one file comfortably
  ([above](#several-agents-on-one-database)), but this is not built for a
  multi-tenant *server* write load.

## License

MIT
