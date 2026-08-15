"""CLI for ingesting and testing the store without going through Claude.

    ctxdb collection create project --embeddings local
    ctxdb ingest project ./docs --pattern "*.md"
    ctxdb fact project "The store is SQLite with sqlite-vec" --key arch.db
    ctxdb search project "which search engine do we use"
    ctxdb status
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import db, retrieve, store
from .embeddings import DEFAULT_LOCAL_MODEL, DEFAULT_VOYAGE_MODEL

ALIAS = {"local": f"local:{DEFAULT_LOCAL_MODEL}", "voyage": f"voyage:{DEFAULT_VOYAGE_MODEL}"}


def _out(value) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ctxdb", description=__doc__)
    parser.add_argument("--db", help="path to the .db file")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_col = sub.add_parser("collection", help="manage collections")
    col_sub = p_col.add_subparsers(dest="sub", required=True)
    p_new = col_sub.add_parser("create")
    p_new.add_argument("name")
    p_new.add_argument("--embeddings", default="none", help="none | local | voyage | <spec>")
    p_new.add_argument("--description", default=None)
    col_sub.add_parser("list")

    p_ing = sub.add_parser("ingest", help="index a file or a directory")
    p_ing.add_argument("collection")
    p_ing.add_argument("path")
    p_ing.add_argument("--pattern", default="*.md")
    p_ing.add_argument("--kind", default=None)

    p_fact = sub.add_parser("fact", help="set an atomic claim")
    p_fact.add_argument("collection")
    p_fact.add_argument("statement")
    p_fact.add_argument("--key", default=None)
    p_fact.add_argument("--subject", default=None)

    p_rel = sub.add_parser("relate", help="connect two entities")
    p_rel.add_argument("collection")
    p_rel.add_argument("source")
    p_rel.add_argument("type")
    p_rel.add_argument("target")

    p_search = sub.add_parser("search", help="hybrid search")
    p_search.add_argument("collection")
    p_search.add_argument("query")
    p_search.add_argument("-k", type=int, default=8)
    p_search.add_argument("--tokens", type=int, default=1500)
    p_search.add_argument("--kinds", default=None)
    p_search.add_argument("--entities", default=None)
    p_search.add_argument("--neighbors", type=int, default=0)
    p_search.add_argument("--json", action="store_true")

    p_ent = sub.add_parser("entity", help="entity card")
    p_ent.add_argument("collection")
    p_ent.add_argument("name")

    sub.add_parser("status", help="inventory of the store")
    sub.add_parser("serve", help="start the MCP server (stdio)")

    args = parser.parse_args(argv)

    if args.cmd == "serve":
        from .server import main as serve

        serve()
        return 0

    conn = db.connect(args.db)

    if args.cmd == "collection" and args.sub == "create":
        spec = ALIAS.get(args.embeddings, args.embeddings)
        row = store.get_or_create_collection(conn, args.name, spec, args.description)
        _out({k: row[k] for k in row.keys()})

    elif args.cmd == "collection":
        _out(store.list_collections(conn))

    elif args.cmd == "ingest":
        path = Path(args.path)
        files = sorted(path.rglob(args.pattern)) if path.is_dir() else [path]
        total = 0
        for file in files:
            result = store.add_document(
                conn,
                args.collection,
                file.read_text(encoding="utf-8", errors="replace"),
                uri=str(file.as_posix()),
                title=file.stem,
                kind=args.kind,
            )
            total += result["chunks"]
            print(f"{file}: {result['chunks']} chunks ({result['status']})", file=sys.stderr)
        _out({"files": len(files), "chunks": total})

    elif args.cmd == "fact":
        _out(
            store.set_fact(
                conn, args.collection, args.statement, key=args.key, subject=args.subject
            )
        )

    elif args.cmd == "relate":
        _out(store.add_relation(conn, args.collection, args.source, args.type, args.target))

    elif args.cmd == "search":
        result = retrieve.search(
            conn,
            args.collection,
            args.query,
            k=args.k,
            budget_tokens=args.tokens or None,
            kinds=args.kinds.split(",") if args.kinds else None,
            entities=args.entities.split(",") if args.entities else None,
            neighbors=args.neighbors,
        )
        _out(result) if args.json else print(retrieve.render_context(result))

    elif args.cmd == "entity":
        _out(retrieve.recall_entity(conn, args.collection, args.name))

    elif args.cmd == "status":
        _out(store.stats(conn))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
