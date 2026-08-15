"""Connection handling, extensions and migrations."""

from __future__ import annotations

import os
import sqlite3
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from .schema import DDL, SCHEMA_VERSION, VEC_TABLE_DDL

DEFAULT_DB_PATH = Path(os.environ.get("CTXDB_PATH", Path.home() / ".ctxdb" / "context.db"))

# How long a writer waits for the lock before giving up, in milliseconds.
#
# Python's sqlite3 defaults to 5 s, which is fine for one process and thin for
# several: with two agents working at once, an ingest that takes longer than that
# makes every other write fail outright rather than queue behind it. SQLite only
# ever admits one writer, so waiting is the correct behaviour — the alternative is
# not more concurrency, it is a lost write.
DEFAULT_BUSY_TIMEOUT_MS = int(os.environ.get("CTXDB_BUSY_TIMEOUT", "20000"))

# Which agent is writing. Recorded on everything it stores.
#
# One database serves several coding agents at once — Claude Code, Cocos, whatever
# comes next — and once they share it, "who wrote this" stops being a curiosity: it
# is how you audit a wrong answer back to the session that planted it.
CLIENT = os.environ.get("CTXDB_CLIENT", "unknown")


def now() -> str:
    """ISO-8601 UTC timestamp. Everything in the database uses this format."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize(text: str) -> str:
    """Lowercased and unaccented: the canonical form used to dedupe entities."""
    decomposed = unicodedata.normalize("NFD", text.strip().lower())
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


def estimate_tokens(text: str) -> int:
    """Cheap approximation. Prose tokenizes at roughly 3.7 characters per token."""
    return max(1, round(len(text) / 3.7))


class Connection(sqlite3.Connection):
    """sqlite3.Connection rejects custom attributes; this subclass accepts them,
    so vector-extension availability travels with the connection."""

    vec_available: bool = False


def connect(path: str | Path | None = None) -> Connection:
    """Open the database, load sqlite-vec, and apply the schema if needed."""
    db_path = Path(path) if path else DEFAULT_DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # check_same_thread=False because the MCP server runs tools in a thread pool;
    # access is serialized with a lock in server.py.
    conn = sqlite3.connect(
        str(db_path),
        check_same_thread=False,
        factory=Connection,
        timeout=DEFAULT_BUSY_TIMEOUT_MS / 1000,
    )
    conn.row_factory = sqlite3.Row

    # WAL already lets readers work while one writer holds the lock; NORMAL is its
    # matching durability setting. Under WAL a crash can only lose the last commits,
    # never corrupt the file, and in exchange every write stops waiting on an fsync —
    # which is time the lock is not held, and therefore time other agents are not
    # blocked. FULL buys durability that matters for a ledger, not for a cache of
    # context that can be re-ingested.
    conn.execute("PRAGMA synchronous=NORMAL")

    conn.vec_available = _load_sqlite_vec(conn)
    conn.executescript(DDL)
    _migrate(conn)

    conn.execute(
        "INSERT INTO meta (key, value) VALUES ('schema_version', ?)"
        " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Bring an older database up to the current schema.

    `CREATE TABLE IF NOT EXISTS` in the DDL only ever helps a fresh file: a database
    created before a column existed keeps its old shape forever, and the first insert
    naming that column fails. Columns are added one by one because SQLite's `ADD COLUMN`
    is instant and non-destructive, which makes this safe to run on every open.

    Indexes over a migrated column belong **here and not in the DDL**, and that is not a
    matter of taste: the DDL runs first, so an index declared there would be created over
    a column the old database has not got yet, and opening it would fail outright. Every
    existing database would break on upgrade.
    """
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(items)")}
    if "client" not in columns:
        conn.execute("ALTER TABLE items ADD COLUMN client TEXT")

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_items_client ON items(collection_id, client)"
    )


def _load_sqlite_vec(conn: sqlite3.Connection) -> bool:
    """Load the vector extension. Without it the system still works, but with
    lexical search only."""
    try:
        import sqlite_vec
    except ImportError:
        return False
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        return True
    except (AttributeError, sqlite3.OperationalError):
        return False
    finally:
        try:
            conn.enable_load_extension(False)
        except AttributeError:
            pass


def ensure_vec_table(conn: sqlite3.Connection, dim: int) -> str:
    """Create (if missing) the vector table for that dimension and return its name."""
    if not getattr(conn, "vec_available", False):
        raise RuntimeError(
            "sqlite-vec is not available. Install it with `uv pip install sqlite-vec`, "
            "or use a collection with embed_spec='none' (BM25 only)."
        )
    if dim <= 0:
        raise ValueError(f"invalid embedding dimension: {dim}")
    table = f"vec_items_{dim}"
    conn.executescript(VEC_TABLE_DDL.format(dim=dim))
    return table
