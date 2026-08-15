"""Hybrid retrieval: BM25 + vectors, fused and packed to a token budget.

Why hybrid and not vectors alone: embeddings are good at paraphrase ("how do I
void an invoice" ~ "credit note procedure") and bad at literals (a SKU, an error
code, a surname). BM25 is exactly the opposite. Fusing both rankings recovers
what each one sees on its own.

The fusion is RRF (Reciprocal Rank Fusion): it sums 1/(k+position) from every
ranking. It works on positions, not scores, which is why nothing has to be
normalized between a cosine distance and a BM25 score — two scales that are not
comparable in the first place.

The last step is the one almost nobody implements and the one that pays off
most: packing to a token budget. Returning 20 chunks "just in case" dilutes the
model's attention; you get what fits, ordered by relevance.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from .db import normalize
from .embeddings import get_embedder
from .store import require_collection

RRF_K = 60  # standard damping: keeps rank 1 from crushing everything below it

KIND_WEIGHT = {
    "fact": 1.20,  # a curated claim is worth more than a raw paragraph
    "note": 1.05,
    "chunk": 1.00,
}

# Both languages on purpose: the FTS tokenizer is accent-insensitive, so one
# collection can hold mixed-language material.
STOPWORDS = {
    "de", "la", "el", "en", "y", "a", "los", "las", "un", "una", "del", "que",
    "por", "con", "para", "es", "se", "su", "al", "lo", "como", "mas", "más",
    "the", "of", "and", "to", "in", "is", "it", "for", "on", "with", "how",
    "what", "which", "when", "where", "who", "why", "does", "do", "can",
}


@dataclass
class Hit:
    id: int
    kind: str
    text: str
    title: str | None
    score: float
    source: str | None
    created_at: str
    tokens: int
    signals: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "text": self.text,
            "score": round(self.score, 5),
            "source": self.source,
            "created_at": self.created_at,
            "tokens": self.tokens,
            "signals": self.signals,
            "meta": self.meta,
        }


# --------------------------------------------------------------------------
# Search branches
# --------------------------------------------------------------------------


def build_fts_query(query: str) -> str:
    """Turn natural language into a safe FTS5 expression.

    Every term is quoted (so a user's `-` or `:` is never read as syntax) and
    the terms are OR-ed: BM25 already rewards documents that concentrate several
    of them.
    """
    terms = [t for t in re.findall(r"[\w\-\.]+", query, flags=re.UNICODE) if len(t) > 1]
    kept = [t for t in terms if normalize(t) not in STOPWORDS] or terms
    return " OR ".join(f'"{t}"' for t in kept)


def _live_clause(include_superseded: bool) -> str:
    if include_superseded:
        return ""
    # A superseded or expired fact stays in the database for auditing, but must
    # never re-enter the context of an answer.
    return " AND i.superseded_by IS NULL AND (i.valid_until IS NULL OR i.valid_until > datetime('now'))"


def lexical_search(
    conn: sqlite3.Connection,
    collection: sqlite3.Row,
    query: str,
    limit: int,
    kinds: list[str] | None = None,
    include_superseded: bool = False,
) -> list[int]:
    match = build_fts_query(query)
    if not match:
        return []
    sql = (
        "SELECT f.rowid AS id FROM items_fts f JOIN items i ON i.id = f.rowid"
        " WHERE items_fts MATCH ? AND i.collection_id = ?"
    )
    params: list[Any] = [match, collection["id"]]
    if kinds:
        sql += f" AND i.kind IN ({','.join('?' * len(kinds))})"
        params.extend(kinds)
    sql += _live_clause(include_superseded)
    # The title (heading path / fact subject) is weighted above the body.
    sql += " ORDER BY bm25(items_fts, 1.0, 1.8) LIMIT ?"
    params.append(limit)
    try:
        return [r["id"] for r in conn.execute(sql, params)]
    except sqlite3.OperationalError:
        return []  # a query FTS5 could not parse: skip the lexical branch


def vector_search(
    conn: sqlite3.Connection,
    collection: sqlite3.Row,
    query: str,
    limit: int,
    kinds: list[str] | None = None,
    include_superseded: bool = False,
) -> list[int]:
    dim = collection["embed_dim"]
    if dim <= 0 or not getattr(conn, "vec_available", False):
        return []
    table = f"vec_items_{dim}"
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (table,)).fetchone():
        return []

    import sqlite_vec

    vectors = get_embedder(collection["embed_spec"]).embed([query], mode="query")
    if not vectors:
        return []

    # vec0 filters neither by collection nor by validity, so over-fetch and
    # discard afterwards. A factor of 6 covers databases with several
    # collections interleaved.
    knn = conn.execute(
        f"SELECT rowid AS id, distance FROM {table} WHERE embedding MATCH ? AND k = ?",
        (sqlite_vec.serialize_float32(vectors[0]), max(limit * 6, 40)),
    ).fetchall()
    if not knn:
        return []

    ids = [r["id"] for r in knn]
    sql = (
        f"SELECT i.id FROM items i WHERE i.id IN ({','.join('?' * len(ids))})"
        " AND i.collection_id = ?"
    )
    params: list[Any] = [*ids, collection["id"]]
    if kinds:
        sql += f" AND i.kind IN ({','.join('?' * len(kinds))})"
        params.extend(kinds)
    sql += _live_clause(include_superseded)
    allowed = {r["id"] for r in conn.execute(sql, params)}

    return [r["id"] for r in knn if r["id"] in allowed][:limit]


def entity_search(
    conn: sqlite3.Connection,
    collection: sqlite3.Row,
    entity_names: list[str],
    limit: int,
    include_superseded: bool = False,
) -> list[int]:
    """Items linked to the given entities, plus those of their graph neighbours.

    This is what makes "what do I know about X?" answerable with material that
    never names X literally but hangs off something related to it.
    """
    norms = [normalize(n) for n in entity_names if n and n.strip()]
    if not norms:
        return []
    seeds = [
        r["id"]
        for r in conn.execute(
            f"SELECT id FROM entities WHERE collection_id = ? AND norm_name IN"
            f" ({','.join('?' * len(norms))})",
            (collection["id"], *norms),
        )
    ]
    if not seeds:
        return []

    neighbors = [
        r["id"]
        for r in conn.execute(
            f"SELECT CASE WHEN src_id IN ({','.join('?' * len(seeds))}) THEN dst_id"
            f" ELSE src_id END AS id FROM relations WHERE collection_id = ?"
            f" AND (src_id IN ({','.join('?' * len(seeds))})"
            f"      OR dst_id IN ({','.join('?' * len(seeds))}))",
            (*seeds, collection["id"], *seeds, *seeds),
        )
    ]
    entity_ids = list(dict.fromkeys(seeds + neighbors))

    sql = (
        f"SELECT i.id FROM items i JOIN item_entities ie ON ie.item_id = i.id"
        f" WHERE ie.entity_id IN ({','.join('?' * len(entity_ids))}) AND i.collection_id = ?"
    )
    params: list[Any] = [*entity_ids, collection["id"]]
    sql += _live_clause(include_superseded)
    # Items linked to a seed entity first, everything else after.
    sql += (
        f" ORDER BY (ie.entity_id IN ({','.join('?' * len(seeds))})) DESC,"
        " i.confidence DESC, i.created_at DESC LIMIT ?"
    )
    params.extend([*seeds, limit])
    return list(dict.fromkeys(r["id"] for r in conn.execute(sql, params)))


# --------------------------------------------------------------------------
# Fusion and packing
# --------------------------------------------------------------------------


def rrf_fuse(rankings: dict[str, list[int]], k: int = RRF_K) -> dict[int, tuple[float, list[str]]]:
    fused: dict[int, tuple[float, list[str]]] = {}
    for name, ids in rankings.items():
        for position, item_id in enumerate(ids):
            score, signals = fused.get(item_id, (0.0, []))
            fused[item_id] = (score + 1.0 / (k + position + 1), signals + [name])
    return fused


def search(
    conn: sqlite3.Connection,
    collection_name: str,
    query: str,
    k: int = 8,
    budget_tokens: int | None = 1500,
    kinds: list[str] | None = None,
    entities: list[str] | None = None,
    include_superseded: bool = False,
    neighbors: int = 0,
) -> dict[str, Any]:
    """Hybrid search with a token budget.

    `neighbors=1` attaches the preceding and following chunk of every result
    that came from a document: cheap, and it fixes the classic case of
    retrieving the paragraph where an explanation starts but ends in the next.
    """
    collection = require_collection(conn, collection_name)
    pool = max(k * 4, 20)

    rankings = {
        "lexical": lexical_search(conn, collection, query, pool, kinds, include_superseded),
        "vector": vector_search(conn, collection, query, pool, kinds, include_superseded),
    }
    if entities:
        rankings["entity"] = entity_search(conn, collection, entities, pool, include_superseded)

    fused = rrf_fuse({name: ids for name, ids in rankings.items() if ids})
    if not fused:
        return {"query": query, "collection": collection_name, "hits": [], "tokens": 0}

    rows = {
        r["id"]: r
        for r in conn.execute(
            f"SELECT i.*, s.uri AS source_uri FROM items i"
            f" LEFT JOIN sources s ON s.id = i.source_id"
            f" WHERE i.id IN ({','.join('?' * len(fused))})",
            tuple(fused),
        )
    }

    hits: list[Hit] = []
    for item_id, (score, signals) in fused.items():
        row = rows.get(item_id)
        if row is None:
            continue
        # An item present in both rankings already scored twice through RRF;
        # these factors only break ties between near-equals.
        adjusted = score * KIND_WEIGHT.get(row["kind"], 1.0) * (0.5 + 0.5 * row["confidence"])
        hits.append(
            Hit(
                id=item_id,
                kind=row["kind"],
                text=row["text"],
                title=row["title"],
                score=adjusted,
                source=row["source_uri"],
                created_at=row["created_at"],
                tokens=row["token_estimate"],
                signals=sorted(set(signals)),
                meta=json.loads(row["meta"] or "{}"),
            )
        )

    hits.sort(key=lambda h: h.score, reverse=True)
    hits = hits[:k]

    if neighbors > 0:
        hits = _expand_neighbors(conn, collection, hits, neighbors)

    if budget_tokens:
        packed: list[Hit] = []
        used = 0
        for hit in hits:
            if used + hit.tokens > budget_tokens and packed:
                continue  # skip what does not fit, but keep trying shorter ones
            packed.append(hit)
            used += hit.tokens
        hits = packed

    return {
        "query": query,
        "collection": collection_name,
        "hits": [h.to_dict() for h in hits],
        "tokens": sum(h.tokens for h in hits),
        "branches": {name: len(ids) for name, ids in rankings.items()},
    }


def _expand_neighbors(
    conn: sqlite3.Connection, collection: sqlite3.Row, hits: list[Hit], window: int
) -> list[Hit]:
    """Attach the adjacent chunks of the same document to a hit's text."""
    for hit in hits:
        if hit.kind != "chunk" or hit.source is None:
            continue
        row = conn.execute(
            "SELECT source_id, ord FROM items WHERE id = ?", (hit.id,)
        ).fetchone()
        if row is None or row["ord"] is None:
            continue
        around = conn.execute(
            "SELECT ord, text FROM items WHERE source_id = ? AND ord BETWEEN ? AND ?"
            " AND id != ? ORDER BY ord",
            (row["source_id"], row["ord"] - window, row["ord"] + window, hit.id),
        ).fetchall()
        if not around:
            continue
        before = [r["text"] for r in around if r["ord"] < row["ord"]]
        after = [r["text"] for r in around if r["ord"] > row["ord"]]
        hit.text = "\n\n".join([*before, hit.text, *after])
        hit.tokens = sum(len(t) for t in [hit.text]) // 4 or hit.tokens
    return hits


def recall_entity(
    conn: sqlite3.Connection, collection_name: str, name: str, k: int = 10
) -> dict[str, Any]:
    """Full card for an entity: summary, relations and linked material."""
    collection = require_collection(conn, collection_name)
    entity = conn.execute(
        "SELECT * FROM entities WHERE collection_id = ? AND norm_name = ?",
        (collection["id"], normalize(name)),
    ).fetchone()
    if entity is None:
        return {"entity": None, "suggestions": _suggest_entities(conn, collection, name)}

    relations = [
        dict(r)
        for r in conn.execute(
            "SELECT r.type, e2.name AS other, r.weight,"
            "       CASE WHEN r.src_id = ? THEN 'outgoing' ELSE 'incoming' END AS direction"
            " FROM relations r"
            " JOIN entities e2 ON e2.id = CASE WHEN r.src_id = ? THEN r.dst_id ELSE r.src_id END"
            " WHERE r.src_id = ? OR r.dst_id = ?"
            " ORDER BY r.weight DESC",
            (entity["id"], entity["id"], entity["id"], entity["id"]),
        )
    ]

    item_ids = entity_search(conn, collection, [entity["name"]], k)
    items = [
        dict(r)
        for r in conn.execute(
            f"SELECT id, kind, title, text, created_at FROM items"
            f" WHERE id IN ({','.join('?' * len(item_ids))})"
            f" ORDER BY CASE kind WHEN 'fact' THEN 0 ELSE 1 END, created_at DESC",
            tuple(item_ids),
        )
    ] if item_ids else []

    return {
        "entity": {
            "name": entity["name"],
            "type": entity["type"],
            "summary": entity["summary"],
        },
        "relations": relations,
        "items": items,
    }


def _suggest_entities(conn: sqlite3.Connection, collection: sqlite3.Row, name: str) -> list[str]:
    norm = normalize(name)
    return [
        r["name"]
        for r in conn.execute(
            "SELECT name FROM entities WHERE collection_id = ? AND norm_name LIKE ? LIMIT 5",
            (collection["id"], f"%{norm}%"),
        )
    ]


def render_context(result: dict[str, Any]) -> str:
    """Format results as a block ready to inject into a prompt.

    Every chunk is labelled with its origin so the model can cite where each
    piece came from, and so you can tell when it answered with something that
    was not in the retrieved context at all.
    """
    hits = result.get("hits", [])
    if not hits:
        return f"<context query=\"{result.get('query', '')}\">\n(no results)\n</context>"

    lines = [f'<context query="{result.get("query", "")}" chunks="{len(hits)}">']
    for i, hit in enumerate(hits, 1):
        origin = hit.get("source") or hit.get("title") or hit["kind"]
        lines.append(f'\n[{i}] ({hit["kind"]}) {origin}')
        if hit.get("title") and hit.get("source"):
            lines.append(f'    section: {hit["title"]}')
        lines.append(hit["text"].strip())
    lines.append("\n</context>")
    return "\n".join(lines)
