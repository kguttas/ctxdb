"""ctxdb — a context database for LLMs.

Minimal usage:

    from ctxdb import connect, get_or_create_collection, add_document, search, render_context

    conn = connect("context.db")
    get_or_create_collection(conn, "project", embed_spec="local:intfloat/multilingual-e5-small")
    add_document(conn, "project", open("manual.md").read(), uri="manual.md")
    print(render_context(search(conn, "project", "how do I void an invoice", budget_tokens=1200)))

Several processes may share one file — that is how more than one coding agent uses the
same memory. Give each its own `connect()`; do not pass a connection between processes.
Set `CTXDB_CLIENT` so what each writes can be told apart later.
"""

from .chunking import chunk_text
from .db import connect, estimate_tokens, normalize, now
from .retrieve import recall_entity, render_context, search
from .store import (
    add_document,
    add_note,
    add_relation,
    delete_source,
    forget,
    get_or_create_collection,
    list_collections,
    set_fact,
    stats,
    upsert_entity,
)

__version__ = "0.1.0"

__all__ = [
    "connect",
    "now",
    "normalize",
    "estimate_tokens",
    "chunk_text",
    "get_or_create_collection",
    "list_collections",
    "add_document",
    "add_note",
    "set_fact",
    "upsert_entity",
    "add_relation",
    "delete_source",
    "forget",
    "stats",
    "search",
    "recall_entity",
    "render_context",
]
