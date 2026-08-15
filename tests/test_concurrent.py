"""Several agents writing to the same database at once.

One `.db` file now serves more than one coding agent — Claude Code, Cocos, whatever
comes next — and they do not take turns. SQLite admits a single writer, so what decides
whether that works is not how many writers there are but **how long each one holds the
lock**.

The failure this guards against was measured, not imagined: embedding used to run inside
the write transaction, so ingesting a document held the lock for as long as the model
took. With the lock held past the busy timeout, a second agent lost every write it
attempted — twelve out of twelve.

Run it directly, like the other suites:

    python tests/test_concurrent.py
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# How long the slow agent pretends its embedding model takes. Comfortably past the
# 5 s that Python's sqlite3 defaults to, so a regression fails loudly instead of
# depending on timing luck.
SLOW_SECONDS = 8.0

WRITER_ROUNDS = 12


def _install_slow_embedder(seconds: float) -> None:
    """Replace the embedder with one that is slow but needs nothing installed.

    A real model would make this suite depend on a download; the point being tested is
    the *duration*, not the arithmetic.
    """
    from ctxdb import store

    class SlowEmbedder:
        spec = "slow:test"
        dim = 4

        def embed(self, texts, mode="document"):  # noqa: ANN001, ANN201
            time.sleep(seconds)
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    store.get_embedder = lambda spec: SlowEmbedder()  # type: ignore[assignment]


def role_ingester(path: str) -> None:
    """The agent that ingests a document with embeddings: the slow one."""
    _install_slow_embedder(SLOW_SECONDS)
    from ctxdb import db, store

    conn = db.connect(path)
    store.get_or_create_collection(conn, "shared", embed_spec="slow:test")
    store.add_document(conn, "shared", "# Manual\n\nA passage worth indexing.\n", uri="manual.md")
    print("ingester: done", flush=True)


def role_writer(path: str) -> None:
    """Another agent doing ordinary short writes while the first one works.

    It shares the collection, and therefore its embedding engine — but its own writes
    are short. That asymmetry is the realistic case: one agent ingesting a manual while
    another jots down a note.
    """
    _install_slow_embedder(0)
    from ctxdb import db, store

    conn = db.connect(path)
    store.get_or_create_collection(conn, "shared", embed_spec="slow:test")

    written, blocked = 0, 0
    for i in range(WRITER_ROUNDS):
        try:
            store.add_note(conn, "shared", f"note {i} from the second agent")
            written += 1
        except sqlite3.OperationalError as err:
            if "locked" in str(err) or "busy" in str(err):
                blocked += 1
            else:
                raise
        time.sleep(0.15)

    print(f"writer: written={written} blocked={blocked}", flush=True)


def _spawn(role: str, path: str) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, __file__, role, path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_two_agents_write_at_the_same_time() -> None:
    # `ignore_cleanup_errors` because Windows refuses to delete a database file whose
    # handle it still considers open for a moment after close. It is a quirk of the
    # temporary directory, not of the store, and failing the suite over it would hide
    # the thing actually being measured.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = str(Path(tmp) / "shared.db")

        # The collection is created up front so neither child races to create it.
        _install_slow_embedder(0)
        from ctxdb import db, store

        arranque = db.connect(path)
        store.get_or_create_collection(arranque, "shared", embed_spec="slow:test")
        arranque.close()

        started = time.monotonic()
        ingester = _spawn("ingester", path)
        time.sleep(0.5)  # let the slow one get going first
        writer = _spawn("writer", path)

        out_i, err_i = ingester.communicate(timeout=120)
        out_w, err_w = writer.communicate(timeout=120)
        elapsed = time.monotonic() - started

        assert ingester.returncode == 0, f"the ingester failed:\n{err_i}"
        assert writer.returncode == 0, f"the writer failed:\n{err_w}"

        blocked = int(out_w.split("blocked=")[1].split()[0])
        written = int(out_w.split("written=")[1].split()[0])

        assert blocked == 0, (
            f"{blocked} writes were rejected while the other agent was ingesting. "
            "Something slow is holding the write lock again — check that embedding "
            "still happens before the transaction opens (store.embed_ahead)."
        )
        assert written == WRITER_ROUNDS, f"only {written} of {WRITER_ROUNDS} got through"

        print(
            f"OK   both agents wrote through a {SLOW_SECONDS:.0f}s ingest "
            f"({written} notes, 0 blocked, {elapsed:.1f}s wall)"
        )


def test_the_writing_agent_is_recorded() -> None:
    """Who wrote what, which is the point of sharing one database."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = str(Path(tmp) / "attributed.db")

        import importlib

        from ctxdb import db, store

        os.environ["CTXDB_CLIENT"] = "cocos"
        importlib.reload(db)
        importlib.reload(store)

        conn = db.connect(path)
        store.get_or_create_collection(conn, "shared", embed_spec="none")
        store.set_fact(conn, "shared", "The engine is llama.cpp", key="arch.engine")
        store.add_note(conn, "shared", "A loose observation")
        store.add_document(conn, "shared", "# Title\n\nBody.\n", uri="doc.md")

        clients = {
            row["client"] for row in conn.execute("SELECT DISTINCT client FROM items")
        }
        assert clients == {"cocos"}, f"expected everything tagged 'cocos', got {clients}"

        by_kind = dict(
            conn.execute("SELECT kind, client FROM items GROUP BY kind").fetchall()
        )
        assert set(by_kind) == {"fact", "note", "chunk"}, by_kind

        # Windows will not delete a file that is still open.
        conn.close()
        del os.environ["CTXDB_CLIENT"]
        importlib.reload(db)
        importlib.reload(store)
        print("OK   every kind of item records which agent wrote it")


def test_an_older_database_gains_the_column() -> None:
    """A database written before attribution existed must keep working."""
    if sqlite3.sqlite_version_info < (3, 35):
        # Only the *simulation* of the old schema needs DROP COLUMN; the migration
        # itself works anywhere. Saying so beats a silent skip.
        print(
            f"SKIP the migration test needs SQLite 3.35 to fake the old schema "
            f"(this one is {sqlite3.sqlite_version})"
        )
        return

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = str(Path(tmp) / "old.db")

        from ctxdb import db, store

        # Build it, then take the column away to imitate the previous schema.
        conn = db.connect(path)
        store.get_or_create_collection(conn, "shared", embed_spec="none")
        store.add_note(conn, "shared", "written by an older version")
        conn.close()

        legacy = sqlite3.connect(path)
        legacy.execute("DROP INDEX IF EXISTS idx_items_client")
        legacy.execute("ALTER TABLE items DROP COLUMN client")
        legacy.commit()
        legacy.close()

        # Reopening must migrate it rather than fail on the first insert.
        conn = db.connect(path)
        store.add_note(conn, "shared", "written after the upgrade")

        rows = conn.execute("SELECT text, client FROM items ORDER BY id").fetchall()
        assert len(rows) == 2, rows
        assert rows[0]["client"] is None, "the old row keeps no attribution, and that is fine"
        assert rows[1]["client"] is not None, "the new one is attributed"
        conn.close()
        print("OK   a database from the previous schema migrates instead of breaking")


if __name__ == "__main__":
    if len(sys.argv) > 1:  # running as one of the child agents
        {"ingester": role_ingester, "writer": role_writer}[sys.argv[1]](sys.argv[2])
        sys.exit(0)

    test_two_agents_write_at_the_same_time()
    test_the_writing_agent_is_recorded()
    test_an_older_database_gains_the_column()
    print("\nConcurrency all green.")
