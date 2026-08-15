"""Write API: collections, documents, facts, and the entity graph."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from typing import Any, Iterable, NamedTuple

from .chunking import chunk_text
from .db import CLIENT, ensure_vec_table, estimate_tokens, normalize, now
from .embeddings import get_embedder


# --------------------------------------------------------------------------
# Collections
# --------------------------------------------------------------------------


def get_or_create_collection(
    conn: sqlite3.Connection,
    name: str,
    embed_spec: str | None = None,
    description: str | None = None,
) -> sqlite3.Row:
    """Return the collection, creating it if it does not exist.

    `embed_spec` is pinned at creation time: changing it later would invalidate
    the vectors already stored, so switching requires a deliberate reindex.
    """
    row = conn.execute("SELECT * FROM collections WHERE name = ?", (name,)).fetchone()
    if row is not None:
        return row

    spec = embed_spec or "none"
    dim = get_embedder(spec).dim
    conn.execute(
        "INSERT INTO collections (name, description, embed_spec, embed_dim, created_at)"
        " VALUES (?, ?, ?, ?, ?)",
        (name, description, spec, dim, now()),
    )
    conn.commit()
    return conn.execute("SELECT * FROM collections WHERE name = ?", (name,)).fetchone()


def require_collection(conn: sqlite3.Connection, name: str) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM collections WHERE name = ?", (name,)).fetchone()
    if row is None:
        known = [r["name"] for r in conn.execute("SELECT name FROM collections")]
        raise KeyError(f"no such collection {name!r}. Available: {known or 'none'}")
    return row


def list_collections(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT c.name, c.description, c.embed_spec, c.embed_dim, c.created_at,
               (SELECT COUNT(*) FROM items i
                 WHERE i.collection_id = c.id AND i.kind = 'chunk')       AS chunks,
               (SELECT COUNT(*) FROM items i
                 WHERE i.collection_id = c.id AND i.kind = 'fact'
                   AND i.superseded_by IS NULL)                           AS live_facts,
               (SELECT COUNT(*) FROM entities e WHERE e.collection_id = c.id) AS entities
          FROM collections c ORDER BY c.name
        """
    ).fetchall()
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------
# Vector indexing
# --------------------------------------------------------------------------


class Embedded(NamedTuple):
    """Vectors already computed, waiting to be written."""

    spec: str
    dim: int
    vectors: list[list[float]]


def embed_ahead(collection: sqlite3.Row, texts: list[str]) -> Embedded | None:
    """Compute the vectors *before* opening the write transaction.

    This split is what makes several agents able to work at once. SQLite admits one
    writer at a time, so the only thing that matters for concurrency is how long that
    writer holds the lock. Embedding inside the transaction held it for as long as the
    model took — or, with a hosted provider, for a network round trip — and every other
    agent's write failed once that went past the busy timeout. Measured: with the lock
    held eight seconds, a second agent lost twelve writes out of twelve.

    Computing first costs nothing when the work turns out to be unnecessary (the caller
    checks for "unchanged" before asking), and moves the slow part to where it belongs:
    outside the lock.
    """
    if not texts or collection["embed_dim"] <= 0:
        return None

    embedder = get_embedder(collection["embed_spec"])
    vectors = embedder.embed(texts, mode="document")
    return Embedded(embedder.spec, embedder.dim, vectors)


def _store_vectors(
    conn: sqlite3.Connection, item_ids: list[int], embedded: Embedded | None
) -> None:
    """Write vectors that are already computed. No model, no network, no waiting."""
    if embedded is None or not item_ids:
        return

    import sqlite_vec

    table = ensure_vec_table(conn, embedded.dim)
    conn.executemany(f"DELETE FROM {table} WHERE rowid = ?", [(i,) for i in item_ids])
    conn.executemany(
        f"INSERT INTO {table} (rowid, embedding) VALUES (?, ?)",
        [(i, sqlite_vec.serialize_float32(v)) for i, v in zip(item_ids, embedded.vectors)],
    )
    conn.executemany(
        "UPDATE items SET embed_model = ? WHERE id = ?",
        [(embedded.spec, i) for i in item_ids],
    )


def _drop_vectors(conn: sqlite3.Connection, dim: int, item_ids: Iterable[int]) -> None:
    ids = list(item_ids)
    if not ids or dim <= 0 or not getattr(conn, "vec_available", False):
        return
    table = f"vec_items_{dim}"
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name = ?", (table,)
    ).fetchone()
    if exists:
        conn.executemany(f"DELETE FROM {table} WHERE rowid = ?", [(i,) for i in ids])


# --------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------


def add_document(
    conn: sqlite3.Connection,
    collection_name: str,
    text: str,
    uri: str | None = None,
    title: str | None = None,
    kind: str | None = None,
    meta: dict[str, Any] | None = None,
    entities: list[str] | None = None,
    target_tokens: int = 350,
    replace: bool = True,
) -> dict[str, Any]:
    """Chunk a document and index it.

    With `replace` set and a document already stored under the same `uri`, the
    old one is swapped out wholesale: re-ingesting a newer version of a file
    leaves no stale chunks contradicting the fresh ones.
    """
    collection = require_collection(conn, collection_name)
    stamp = now()
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    existing = None
    if uri:
        existing = conn.execute(
            "SELECT * FROM sources WHERE collection_id = ? AND uri = ?",
            (collection["id"], uri),
        ).fetchone()

    if existing and existing["content_hash"] == content_hash:
        n = conn.execute(
            "SELECT COUNT(*) AS n FROM items WHERE source_id = ?", (existing["id"],)
        ).fetchone()["n"]
        return {"source_id": existing["id"], "chunks": n, "status": "unchanged"}

    if existing and not replace:
        raise ValueError(f"a document with uri={uri!r} already exists (pass replace=True)")

    # Chunking and embedding happen before anything is written, so the lock is held
    # only for the inserts. See `embed_ahead`: this is what lets a second agent keep
    # working while this one ingests a manual.
    chunks = chunk_text(text, target_tokens=target_tokens)
    embedded = embed_ahead(
        collection, [f"{c.heading_path}\n{c.text}".strip() for c in chunks]
    )

    if existing:
        delete_source(conn, collection_name, uri=uri)

    cur = conn.execute(
        "INSERT INTO sources (collection_id, uri, title, kind, content_hash, meta, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (collection["id"], uri, title, kind, content_hash, json.dumps(meta or {}), stamp),
    )
    source_id = int(cur.lastrowid)

    item_ids: list[int] = []
    for chunk in chunks:
        # A chunk is indexed together with its heading path, so a query about
        # "credit notes" can reach a paragraph that never says those words but
        # hangs under that section. The title is skipped when the document
        # already opens with an equivalent heading, to avoid repeating it.
        parts = chunk.heading_path.split(" > ") if chunk.heading_path else []
        if title and (not parts or normalize(parts[0]) != normalize(title)):
            parts.insert(0, title)
        heading = " > ".join(p for p in parts if p)

        cur = conn.execute(
            "INSERT INTO items (collection_id, kind, text, title, source_id, ord, meta,"
            " token_estimate, client, created_at, updated_at)"
            " VALUES (?, 'chunk', ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                collection["id"],
                chunk.text,
                heading or None,
                source_id,
                chunk.ord,
                json.dumps(meta or {}),
                estimate_tokens(chunk.text),
                CLIENT,
                stamp,
                stamp,
            ),
        )
        item_ids.append(int(cur.lastrowid))

    _store_vectors(conn, item_ids, embedded)

    for item_id in item_ids:
        _attach_entities(conn, collection, item_id, entities)
        autolink_entities(conn, collection, item_id)

    conn.commit()
    return {"source_id": source_id, "chunks": len(item_ids), "status": "indexed"}


def delete_source(conn: sqlite3.Connection, collection_name: str, uri: str) -> int:
    """Delete a document along with all its chunks and vectors."""
    collection = require_collection(conn, collection_name)
    source = conn.execute(
        "SELECT id FROM sources WHERE collection_id = ? AND uri = ?", (collection["id"], uri)
    ).fetchone()
    if source is None:
        return 0
    ids = [
        r["id"] for r in conn.execute("SELECT id FROM items WHERE source_id = ?", (source["id"],))
    ]
    _drop_vectors(conn, collection["embed_dim"], ids)
    conn.execute("DELETE FROM sources WHERE id = ?", (source["id"],))
    conn.commit()
    return len(ids)


# --------------------------------------------------------------------------
# Facts
# --------------------------------------------------------------------------


def set_fact(
    conn: sqlite3.Connection,
    collection_name: str,
    statement: str,
    key: str | None = None,
    subject: str | None = None,
    confidence: float = 1.0,
    meta: dict[str, Any] | None = None,
    entities: list[str] | None = None,
    valid_from: str | None = None,
) -> dict[str, Any]:
    """Record an atomic claim.

    When a `key` is given and a live fact already holds it, the old one is
    marked superseded rather than deleted: it stops being retrieved, but the
    history stays available to audit what was believed and since when.
    """
    collection = require_collection(conn, collection_name)
    stamp = now()

    previous = None
    if key:
        previous = conn.execute(
            "SELECT id, text FROM items WHERE collection_id = ? AND fact_key = ?"
            " AND superseded_by IS NULL",
            (collection["id"], key),
        ).fetchone()
        if previous and previous["text"].strip() == statement.strip():
            return {"item_id": previous["id"], "status": "unchanged"}

    # After ruling out the unchanged case, and before writing anything.
    embedded = embed_ahead(collection, [f"{subject or ''} {statement}".strip()])

    if previous:
        # Free the live-fact unique index by pointing the old row at itself;
        # below it is corrected to the real replacement id.
        conn.execute(
            "UPDATE items SET superseded_by = id, valid_until = ?, updated_at = ?"
            " WHERE id = ?",
            (stamp, stamp, previous["id"]),
        )

    cur = conn.execute(
        "INSERT INTO items (collection_id, kind, text, title, meta, confidence,"
        " token_estimate, fact_key, valid_from, client, created_at, updated_at)"
        " VALUES (?, 'fact', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            collection["id"],
            statement,
            subject,
            json.dumps(meta or {}),
            confidence,
            estimate_tokens(statement),
            key,
            valid_from or stamp,
            CLIENT,
            stamp,
            stamp,
        ),
    )
    item_id = int(cur.lastrowid)

    if previous:
        conn.execute("UPDATE items SET superseded_by = ? WHERE id = ?", (item_id, previous["id"]))
        _drop_vectors(conn, collection["embed_dim"], [previous["id"]])

    _store_vectors(conn, [item_id], embedded)
    _attach_entities(conn, collection, item_id, entities)
    if subject:
        _attach_entities(conn, collection, item_id, [subject])
    autolink_entities(conn, collection, item_id)

    conn.commit()
    return {
        "item_id": item_id,
        "status": "superseded" if previous else "created",
        "superseded_item": previous["id"] if previous else None,
    }


def add_note(
    conn: sqlite3.Connection,
    collection_name: str,
    text: str,
    title: str | None = None,
    meta: dict[str, Any] | None = None,
    entities: list[str] | None = None,
) -> dict[str, Any]:
    """Free-standing text with no source document. Chunked only if long."""
    collection = require_collection(conn, collection_name)
    stamp = now()
    pieces = chunk_text(text) if estimate_tokens(text) > 600 else None
    bodies = [c.text for c in pieces] if pieces else [text]

    # Before the first insert, so the lock is not held while the model runs.
    embedded = embed_ahead(collection, [f"{title or ''}\n{b}".strip() for b in bodies])

    item_ids: list[int] = []
    for chunk in pieces or [None]:
        body = chunk.text if chunk else text
        cur = conn.execute(
            "INSERT INTO items (collection_id, kind, text, title, ord, meta,"
            " token_estimate, client, created_at, updated_at)"
            " VALUES (?, 'note', ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                collection["id"],
                body,
                title,
                chunk.ord if chunk else 0,
                json.dumps(meta or {}),
                estimate_tokens(body),
                CLIENT,
                stamp,
                stamp,
            ),
        )
        item_ids.append(int(cur.lastrowid))

    _store_vectors(conn, item_ids, embedded)
    for item_id in item_ids:
        _attach_entities(conn, collection, item_id, entities)
        autolink_entities(conn, collection, item_id)

    conn.commit()
    return {"item_ids": item_ids, "chunks": len(item_ids)}


def forget(conn: sqlite3.Connection, collection_name: str, item_ids: list[int]) -> int:
    """Permanently delete specific items, vectors included."""
    collection = require_collection(conn, collection_name)
    ids = [
        r["id"]
        for r in conn.execute(
            f"SELECT id FROM items WHERE collection_id = ? AND id IN"
            f" ({','.join('?' * len(item_ids))})",
            (collection["id"], *item_ids),
        )
    ] if item_ids else []
    if not ids:
        return 0
    _drop_vectors(conn, collection["embed_dim"], ids)
    conn.executemany("DELETE FROM items WHERE id = ?", [(i,) for i in ids])
    conn.commit()
    return len(ids)


# --------------------------------------------------------------------------
# Entity graph
# --------------------------------------------------------------------------


def upsert_entity(
    conn: sqlite3.Connection,
    collection_name: str,
    name: str,
    type: str | None = None,
    summary: str | None = None,
) -> int:
    collection = require_collection(conn, collection_name)
    return _upsert_entity_row(conn, collection, name, type, summary)


def _upsert_entity_row(
    conn: sqlite3.Connection,
    collection: sqlite3.Row,
    name: str,
    type: str | None = None,
    summary: str | None = None,
) -> int:
    norm = normalize(name)
    row = conn.execute(
        "SELECT id FROM entities WHERE collection_id = ? AND norm_name = ?",
        (collection["id"], norm),
    ).fetchone()
    if row:
        if type or summary:
            conn.execute(
                "UPDATE entities SET type = COALESCE(?, type), summary = COALESCE(?, summary)"
                " WHERE id = ?",
                (type, summary, row["id"]),
            )
        return int(row["id"])
    cur = conn.execute(
        "INSERT INTO entities (collection_id, name, norm_name, type, summary, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (collection["id"], name.strip(), norm, type, summary, now()),
    )
    return int(cur.lastrowid)


def add_relation(
    conn: sqlite3.Connection,
    collection_name: str,
    src: str,
    type: str,
    dst: str,
    weight: float = 1.0,
    evidence_item_id: int | None = None,
) -> dict[str, Any]:
    """Link two entities. Entities are created if they did not exist."""
    collection = require_collection(conn, collection_name)
    src_id = _upsert_entity_row(conn, collection, src)
    dst_id = _upsert_entity_row(conn, collection, dst)
    conn.execute(
        "INSERT INTO relations (collection_id, src_id, dst_id, type, weight, item_id, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)"
        " ON CONFLICT (collection_id, src_id, dst_id, type)"
        " DO UPDATE SET weight = excluded.weight",
        (collection["id"], src_id, dst_id, type, weight, evidence_item_id, now()),
    )
    conn.commit()
    return {"src_id": src_id, "dst_id": dst_id, "type": type}


def _attach_entities(
    conn: sqlite3.Connection,
    collection: sqlite3.Row,
    item_id: int,
    names: list[str] | None,
) -> None:
    for name in names or []:
        if not name or not name.strip():
            continue
        entity_id = _upsert_entity_row(conn, collection, name)
        conn.execute(
            "INSERT OR IGNORE INTO item_entities (item_id, entity_id) VALUES (?, ?)",
            (item_id, entity_id),
        )


def autolink_entities(conn: sqlite3.Connection, collection: sqlite3.Row, item_id: int) -> int:
    """Link the item to the *already known* entities it mentions.

    It deliberately does not extract new entities: guessing them with heuristics
    fills the graph with noise. Only entities somebody declared are recognized
    here, which keeps precision high.
    """
    row = conn.execute("SELECT text, title FROM items WHERE id = ?", (item_id,)).fetchone()
    if row is None:
        return 0
    haystack = normalize(f"{row['title'] or ''} {row['text']}")
    linked = 0
    for entity in conn.execute(
        "SELECT id, norm_name FROM entities WHERE collection_id = ?", (collection["id"],)
    ).fetchall():
        name = entity["norm_name"]
        if len(name) < 3:
            continue
        if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", haystack):
            cur = conn.execute(
                "INSERT OR IGNORE INTO item_entities (item_id, entity_id) VALUES (?, ?)",
                (item_id, entity["id"]),
            )
            linked += cur.rowcount or 0
    return linked


def stats(conn: sqlite3.Connection) -> dict[str, Any]:
    return {
        "collections": list_collections(conn),
        "vectors_available": bool(getattr(conn, "vec_available", False)),
        "total_items": conn.execute("SELECT COUNT(*) AS n FROM items").fetchone()["n"],
        "relations": conn.execute("SELECT COUNT(*) AS n FROM relations").fetchone()["n"],
    }
