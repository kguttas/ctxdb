"""Exercises the vector branch (sqlite-vec) without downloading a real model.

It uses a deterministic toy embedder: no semantics, but it drives exactly the
same plumbing a real model would — serialization, the vec0 table, the KNN query,
collection filtering and RRF fusion. If this passes, switching `embed_spec` to
`local:...` or `voyage:...` only changes where the numbers come from.
"""

from __future__ import annotations

import hashlib
import math
import tempfile
from pathlib import Path

from ctxdb import connect, embeddings, get_or_create_collection, search, set_fact, store
from ctxdb.db import ensure_vec_table

DIM = 64


class ToyEmbedder:
    """Bag of words hashed into a 64-dim unit vector. Two texts sharing
    vocabulary end up close, which is all this test needs."""

    spec = "toy:64"
    dim = DIM

    def embed(self, texts, mode="document"):
        out = []
        for text in texts:
            vec = [0.0] * DIM
            for word in text.lower().split():
                h = int(hashlib.sha1(word.encode()).hexdigest(), 16)
                vec[h % DIM] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


def check(condition: bool, label: str) -> None:
    print(f"{'OK  ' if condition else 'FAIL'} {label}")
    if not condition:
        raise AssertionError(label)


def main() -> None:
    embeddings._CACHE["toy:64"] = ToyEmbedder()

    tmp = Path(tempfile.mkdtemp()) / "vec.db"
    conn = connect(tmp)
    check(conn.vec_available, "the sqlite-vec extension loads")

    table = ensure_vec_table(conn, DIM)
    check(table == f"vec_items_{DIM}", "the vector table is created per dimension")

    get_or_create_collection(conn, "vec", embed_spec="toy:64")
    row = conn.execute("SELECT embed_dim FROM collections WHERE name = 'vec'").fetchone()
    check(row["embed_dim"] == DIM, "the collection records the model dimension")

    store.add_document(
        conn,
        "vec",
        "# Deployment\n\nThe service runs in containers orchestrated by Kubernetes.\n\n"
        "## Backups\n\nBackups are stored encrypted in cold storage.",
        uri="ops.md",
        title="Operations",
    )
    set_fact(conn, "vec", "Backups run every night at 02:00.", key="ops.backups.schedule")

    indexed = conn.execute(f"SELECT COUNT(*) AS n FROM vec_items_{DIM}").fetchone()["n"]
    items = conn.execute("SELECT COUNT(*) AS n FROM items").fetchone()["n"]
    check(indexed == items, f"every item has a vector ({indexed}/{items})")
    check(
        conn.execute("SELECT COUNT(*) AS n FROM items WHERE embed_model IS NULL").fetchone()["n"]
        == 0,
        "every item records which model indexed it",
    )

    result = search(conn, "vec", "encrypted backups cold storage")
    check(bool(result["hits"]), "hybrid search returns results")
    check(result["branches"]["vector"] > 0, "the vector branch contributes candidates")
    check(
        any("vector" in h["signals"] for h in result["hits"]),
        "at least one result arrives through the semantic path",
    )

    # Another collection in the same file must not leak into the results.
    get_or_create_collection(conn, "other", embed_spec="toy:64")
    store.add_document(conn, "other", "Backups in the other collection are different.",
                       uri="other.md")
    isolated = search(conn, "vec", "backups")
    check(
        all("other.md" != h["source"] for h in isolated["hits"]),
        "vectors from another collection stay out of the result",
    )

    # Superseding a fact must remove its vector from the index.
    before = conn.execute(f"SELECT COUNT(*) AS n FROM vec_items_{DIM}").fetchone()["n"]
    set_fact(conn, "vec", "Backups run every night at 03:30.", key="ops.backups.schedule")
    after = conn.execute(f"SELECT COUNT(*) AS n FROM vec_items_{DIM}").fetchone()["n"]
    check(after == before, "the superseded fact releases its vector and the new one takes it")

    schedules = [h["text"] for h in search(conn, "vec", "backup schedule")["hits"]]
    check(
        not any("02:00" in t for t in schedules),
        "the stale schedule is no longer retrieved through the semantic path",
    )

    print(f"\nVector branch all green. Test database: {tmp}")


if __name__ == "__main__":
    main()
