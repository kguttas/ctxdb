"""Connection handling, extensions and migrations."""

from __future__ import annotations

import os
import sqlite3
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from .schema import DDL, SCHEMA_VERSION, VEC_TABLE_DDL

DEFAULT_DB_PATH = Path(os.environ.get("CTXDB_PATH", Path.home() / ".ctxdb" / "context.db"))


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
    conn = sqlite3.connect(str(db_path), check_same_thread=False, factory=Connection)
    conn.row_factory = sqlite3.Row

    conn.vec_available = _load_sqlite_vec(conn)
    conn.executescript(DDL)

    version = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    if version is None:
        conn.execute(
            "INSERT INTO meta (key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),)
        )
    conn.commit()
    return conn


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
