"""SQL schema for the context store.

Mental model: everything retrievable lives in `items`, tagged with a `kind` that
says which structural layer it belongs to:

  fact   -> an atomic claim with a validity window. A new fact sharing a
            `fact_key` *supersedes* the old one instead of coexisting with it.
  chunk  -> a passage of a document, with its position and heading path.
  note   -> free-standing text with no source document.

On top sits a lightweight graph (`entities` + `relations`) that answers "what
connects to what" without requiring that information to be present in the text.

Retrieval runs two indexes over those same `items`:
  items_fts        -> FTS5/BM25, unbeatable for names, codes and acronyms.
  vec_items_<dim>  -> sqlite-vec, for semantic similarity.
Vector tables are created per dimension because each collection may use a
different embedding model (384-dim local, 1024-dim Voyage, and so on).
"""

SCHEMA_VERSION = 1

DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- A collection isolates one context domain (a project, a client, a manual).
-- Each one pins its own embedding engine.
CREATE TABLE IF NOT EXISTS collections (
    id           INTEGER PRIMARY KEY,
    name         TEXT NOT NULL UNIQUE,
    description  TEXT,
    embed_spec   TEXT NOT NULL DEFAULT 'none',   -- 'none' | 'local:<model>' | 'voyage:<model>'
    embed_dim    INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL
);

-- Source document. Its `content_hash` lets you re-ingest without duplicating.
CREATE TABLE IF NOT EXISTS sources (
    id            INTEGER PRIMARY KEY,
    collection_id INTEGER NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    uri           TEXT,
    title         TEXT,
    kind          TEXT,
    content_hash  TEXT,
    meta          TEXT NOT NULL DEFAULT '{}',
    created_at    TEXT NOT NULL,
    UNIQUE (collection_id, uri)
);

CREATE TABLE IF NOT EXISTS items (
    id             INTEGER PRIMARY KEY,
    collection_id  INTEGER NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    kind           TEXT NOT NULL CHECK (kind IN ('fact', 'chunk', 'note')),
    text           TEXT NOT NULL,
    title          TEXT,                 -- heading path, or the fact's subject
    source_id      INTEGER REFERENCES sources(id) ON DELETE CASCADE,
    ord            INTEGER,              -- position within the document
    meta           TEXT NOT NULL DEFAULT '{}',   -- free-form JSON: tags, author, ...
    confidence     REAL NOT NULL DEFAULT 1.0,
    token_estimate INTEGER NOT NULL DEFAULT 0,
    fact_key       TEXT,                 -- a fact's stable identity
    valid_from     TEXT,
    valid_until    TEXT,                 -- NULL = still current
    superseded_by  INTEGER REFERENCES items(id) ON DELETE SET NULL,
    embed_model    TEXT,                 -- NULL = no vector indexed
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_items_collection ON items(collection_id, kind);
CREATE INDEX IF NOT EXISTS idx_items_source     ON items(source_id, ord);
CREATE INDEX IF NOT EXISTS idx_items_live       ON items(collection_id, superseded_by, valid_until);

-- At most one live fact per key within a collection.
CREATE UNIQUE INDEX IF NOT EXISTS idx_items_factkey_live
    ON items(collection_id, fact_key)
    WHERE fact_key IS NOT NULL AND superseded_by IS NULL;

CREATE TABLE IF NOT EXISTS entities (
    id            INTEGER PRIMARY KEY,
    collection_id INTEGER NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    name          TEXT NOT NULL,
    norm_name     TEXT NOT NULL,   -- lowercased and unaccented, for deduplication
    type          TEXT,
    summary       TEXT,
    created_at    TEXT NOT NULL,
    UNIQUE (collection_id, norm_name)
);

CREATE TABLE IF NOT EXISTS relations (
    id            INTEGER PRIMARY KEY,
    collection_id INTEGER NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    src_id        INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    dst_id        INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    type          TEXT NOT NULL,
    weight        REAL NOT NULL DEFAULT 1.0,
    item_id       INTEGER REFERENCES items(id) ON DELETE SET NULL,  -- supporting evidence
    created_at    TEXT NOT NULL,
    UNIQUE (collection_id, src_id, dst_id, type)
);

CREATE INDEX IF NOT EXISTS idx_relations_src ON relations(src_id);
CREATE INDEX IF NOT EXISTS idx_relations_dst ON relations(dst_id);

CREATE TABLE IF NOT EXISTS item_entities (
    item_id   INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    PRIMARY KEY (item_id, entity_id)
);

CREATE INDEX IF NOT EXISTS idx_item_entities_entity ON item_entities(entity_id);

-- Lexical index. remove_diacritics=2 makes "diseño" match "diseno", so accented
-- languages behave the way users actually type.
CREATE VIRTUAL TABLE IF NOT EXISTS items_fts USING fts5(
    text,
    title,
    content='items',
    content_rowid='id',
    tokenize="unicode61 remove_diacritics 2 tokenchars '-_.'"
);

CREATE TRIGGER IF NOT EXISTS items_ai AFTER INSERT ON items BEGIN
    INSERT INTO items_fts(rowid, text, title) VALUES (new.id, new.text, new.title);
END;

CREATE TRIGGER IF NOT EXISTS items_ad AFTER DELETE ON items BEGIN
    INSERT INTO items_fts(items_fts, rowid, text, title)
        VALUES ('delete', old.id, old.text, old.title);
END;

CREATE TRIGGER IF NOT EXISTS items_au AFTER UPDATE OF text, title ON items BEGIN
    INSERT INTO items_fts(items_fts, rowid, text, title)
        VALUES ('delete', old.id, old.text, old.title);
    INSERT INTO items_fts(rowid, text, title) VALUES (new.id, new.text, new.title);
END;
"""

# Vector tables are created on demand: one per embedding dimension.
VEC_TABLE_DDL = """
CREATE VIRTUAL TABLE IF NOT EXISTS vec_items_{dim} USING vec0(
    embedding float[{dim}]
);
"""
