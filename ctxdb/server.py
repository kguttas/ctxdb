"""MCP server: exposes the context store as tools for a coding agent.

Setup (Claude Code):

    claude mcp add ctxdb -e CTXDB_CLIENT=claude -- uv --directory /path/to/ctxdb run ctxdb-mcp

Any MCP client works the same way — Cocos and the rest speak the same protocol. Point
them all at one `CTXDB_PATH` and they share memory; give each a different `CTXDB_CLIENT`
and you can still tell who wrote what. Writing at the same time is fine: see
`store.embed_ahead` for what makes that hold.

Environment variables:
    CTXDB_PATH          path to the .db file (default ~/.ctxdb/context.db)
    CTXDB_COLLECTION    default collection when a tool call omits one
    CTXDB_EMBED         embedding spec for the default collection
    CTXDB_CLIENT        name recorded on everything this agent writes
    CTXDB_BUSY_TIMEOUT  milliseconds a writer waits for the lock (default 20000)
    VOYAGE_API_KEY      only if some collection uses embed_spec='voyage:...'
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any

try:  # modern SDK: FastMCP was renamed to MCPServer
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # pragma: no cover - SDK 1.x
    from mcp.server.fastmcp import FastMCP as _Server

from . import db, retrieve, store

mcp = _Server("ctxdb")

_lock = threading.Lock()
_conn = None
DEFAULT_COLLECTION = os.environ.get("CTXDB_COLLECTION", "default")


def _connection():
    global _conn
    if _conn is None:
        _conn = db.connect()
        # Bootstrap collection: without it the first save would fail and Claude
        # would have to guess that it must create one first.
        store.get_or_create_collection(
            _conn,
            DEFAULT_COLLECTION,
            embed_spec=os.environ.get("CTXDB_EMBED", "none"),
            description="Default collection",
        )
    return _conn


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


@mcp.tool()
def context_search(
    query: str,
    collection: str = "",
    k: int = 8,
    budget_tokens: int = 1500,
    kinds: str = "",
    entities: str = "",
    neighbors: int = 0,
    format: str = "text",
) -> str:
    """Retrieve the stored context chunks relevant to a query.

    Combines lexical search (exact: codes, names, acronyms) with semantic search
    (paraphrase), fuses both rankings, and returns only what fits in the given
    token budget.

    Use it BEFORE answering any question about a project, document or detail the
    user stored earlier, instead of assuming you remember it.

    Args:
        query: the question or topic, in natural language.
        collection: domain to search; empty uses the default collection.
        k: maximum number of chunks to consider.
        budget_tokens: ceiling for the returned context; 0 disables packing.
        kinds: comma-separated filter: 'fact', 'chunk', 'note'.
        entities: comma-separated names; also pulls material linked to those
            entities and to their neighbours in the graph.
        neighbors: 1 attaches the adjacent chunks from the same document.
        format: 'text' for a ready-to-read block, 'json' for raw data with
            scores and provenance.
    """
    with _lock:
        result = retrieve.search(
            _connection(),
            collection or DEFAULT_COLLECTION,
            query,
            k=k,
            budget_tokens=budget_tokens or None,
            kinds=[t.strip() for t in kinds.split(",") if t.strip()] or None,
            entities=[e.strip() for e in entities.split(",") if e.strip()] or None,
            neighbors=neighbors,
        )
    return _json(result) if format == "json" else retrieve.render_context(result)


@mcp.tool()
def context_add_document(
    text: str,
    uri: str = "",
    title: str = "",
    collection: str = "",
    kind: str = "",
    entities: str = "",
) -> str:
    """Ingest a whole document: chunk it along its structure and index it.

    For long text (manuals, specs, transcripts, code). Always give a stable
    `uri`: re-ingesting under the same uri replaces the previous version
    wholesale, instead of leaving two versions contradicting each other.

    Args:
        text: full content. Markdown gets the most out of the chunker.
        uri: stable identifier (file path, URL, "meeting-2026-08-14").
        title: human-readable document title.
        collection: domain to store it in.
        kind: free-form label ('manual', 'contract', 'transcript', ...).
        entities: comma-separated names to link to every one of its chunks.
    """
    with _lock:
        result = store.add_document(
            _connection(),
            collection or DEFAULT_COLLECTION,
            text,
            uri=uri or None,
            title=title or None,
            kind=kind or None,
            entities=[e.strip() for e in entities.split(",") if e.strip()] or None,
        )
    return _json(result)


@mcp.tool()
def context_set_fact(
    statement: str,
    key: str = "",
    subject: str = "",
    collection: str = "",
    confidence: float = 1.0,
    entities: str = "",
) -> str:
    """Store an atomic claim that supersedes the previous one under the same key.

    This is the tool for data that CHANGES: decisions, preferences, states,
    versions, owners. Given a stable `key` (for example 'project.database' or
    'client.acme.contact'), the new value displaces the old one and searches
    stop returning the stale version.

    For reference text that does not change, use context_add_document.

    Args:
        statement: a single idea, self-contained and readable without context.
        key: the fact's stable identity; without it facts accumulate instead of
            replacing one another.
        subject: what or who it is about (indexed with extra weight).
        collection: domain to store it in.
        confidence: 0..1; uncertain facts weigh less at retrieval time.
        entities: comma-separated names to link to this fact.
    """
    with _lock:
        result = store.set_fact(
            _connection(),
            collection or DEFAULT_COLLECTION,
            statement,
            key=key or None,
            subject=subject or None,
            confidence=confidence,
            entities=[e.strip() for e in entities.split(",") if e.strip()] or None,
        )
    return _json(result)


@mcp.tool()
def context_add_note(
    text: str, title: str = "", collection: str = "", entities: str = ""
) -> str:
    """Store free-standing text with no source document (an observation, a piece
    of conversation worth remembering). Chunked only if it is long.

    Args:
        text: the content to remember.
        title: short heading that helps retrieve it later.
        collection: domain to store it in.
        entities: comma-separated names to link to the note.
    """
    with _lock:
        result = store.add_note(
            _connection(),
            collection or DEFAULT_COLLECTION,
            text,
            title=title or None,
            entities=[e.strip() for e in entities.split(",") if e.strip()] or None,
        )
    return _json(result)


@mcp.tool()
def context_recall_entity(name: str, collection: str = "", k: int = 10) -> str:
    """Return everything known about an entity: summary, relations and material.

    Use it when the question is about one specific thing (a person, a system, a
    client) rather than about a topic. It also pulls material linked to its
    neighbouring entities, so it recovers content that never names it literally.

    Args:
        name: the entity name; accents and casing do not matter.
        collection: domain to search.
        k: maximum number of items to return.
    """
    with _lock:
        result = retrieve.recall_entity(_connection(), collection or DEFAULT_COLLECTION, name, k)
    return _json(result)


@mcp.tool()
def context_relate(
    source: str, type: str, target: str, collection: str = "", weight: float = 1.0
) -> str:
    """Connect two entities in the graph, creating them if they did not exist.

    Example `type` values: 'depends_on', 'works_at', 'replaces', 'part_of'.
    Relations expand entity searches outward into neighbouring material.

    Args:
        source: starting entity.
        type: relation name, lowercase with underscores.
        target: destination entity.
        collection: domain to record it in.
        weight: 0..1+ to rank relations by importance.
    """
    with _lock:
        result = store.add_relation(
            _connection(), collection or DEFAULT_COLLECTION, source, type, target, weight=weight
        )
    return _json(result)


@mcp.tool()
def context_create_collection(name: str, description: str = "", embeddings: str = "none") -> str:
    """Create an isolated context domain (a project, a client, a manual).

    The embedding engine is pinned at creation time and cannot be changed
    without reindexing, because vectors from different models are not
    comparable.

    Args:
        name: short collection identifier.
        description: what it is for.
        embeddings: 'none' (BM25 only, zero dependencies),
            'local:intfloat/multilingual-e5-small' (offline) or
            'voyage:voyage-3.5' (API, best quality; requires VOYAGE_API_KEY).
    """
    with _lock:
        row = store.get_or_create_collection(
            _connection(), name, embed_spec=embeddings, description=description or None
        )
    return _json({k: row[k] for k in row.keys()})


@mcp.tool()
def context_status() -> str:
    """Inventory of the store: collections, how much each holds, and whether
    vectors are active. Use it to find out where to search before querying."""
    with _lock:
        return _json(store.stats(_connection()))


@mcp.tool()
def context_forget(ids: str, collection: str = "") -> str:
    """Permanently delete items by id (comma-separated).

    For a value that merely changed, prefer context_set_fact with the same key:
    it replaces without losing the history.

    Args:
        ids: numeric ids, comma-separated, as returned by a search.
        collection: domain they belong to.
    """
    item_ids = [int(i) for i in ids.replace(" ", "").split(",") if i]
    with _lock:
        n = store.forget(_connection(), collection or DEFAULT_COLLECTION, item_ids)
    return _json({"deleted": n})


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
