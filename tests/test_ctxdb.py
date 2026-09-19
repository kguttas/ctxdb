"""Engine tests. Run with no embeddings and no network: `python tests/test_ctxdb.py`.

They cover what can actually break silently: that a superseded fact stops being
retrieved, that chunking respects structure, and that lexical search finds what
the semantic branch would never see.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from ctxdb import (
    add_document,
    add_note,
    add_relation,
    chunk_text,
    connect,
    get_or_create_collection,
    recall_entity,
    render_context,
    search,
    set_fact,
    stats,
)
from ctxdb.retrieve import recall_recent, render_recall, search_multi

MANUAL = """# Billing Manual

This manual covers issuing and voiding tax documents.

## Issuing receipts

A receipt is issued from the sales module. The folio number is assigned
automatically from the current authorization batch.

If the batch is exhausted the system blocks issuing and shows error E-4041.
A new batch must be requested from the tax authority portal.

## Credit notes

To reverse an already issued document you issue a credit note referencing the
original folio. There is no direct void operation.

The legal window to issue one is 90 calendar days from the date of the
originating document.

## Technical appendix

```python
def issue(folio: int) -> Document:
    return Document(folio=folio, status="issued")
```
"""


def check(condition: bool, label: str) -> None:
    print(f"{'OK  ' if condition else 'FAIL'} {label}")
    if not condition:
        raise AssertionError(label)


def main() -> None:
    tmp = Path(tempfile.mkdtemp()) / "test.db"
    conn = connect(tmp)
    get_or_create_collection(conn, "test", embed_spec="none", description="tests")

    # --- chunking -------------------------------------------------------
    chunks = chunk_text(MANUAL, target_tokens=80)
    check(len(chunks) >= 4, f"the manual splits into several pieces ({len(chunks)})")
    check(all(c.heading_path for c in chunks[1:]), "every chunk keeps its heading path")
    code_chunks = [c for c in chunks if "def issue" in c.text]
    check(len(code_chunks) == 1, "the code block is not split in two")
    check(
        not any("Credit notes" in c.heading_path and "receipt is issued" in c.text
                for c in chunks),
        "no chunk crosses a heading boundary",
    )

    # --- ingestion ------------------------------------------------------
    result = add_document(conn, "test", MANUAL, uri="manual.md", title="Billing")
    check(result["status"] == "indexed", "the document is indexed")
    again = add_document(conn, "test", MANUAL, uri="manual.md", title="Billing")
    check(again["status"] == "unchanged", "re-ingesting identical content does not duplicate")

    # --- lexical search -------------------------------------------------
    hits = search(conn, "test", "error E-4041 batch exhausted")["hits"]
    check(bool(hits), "finds the chunk by its error code")
    check("E-4041" in hits[0]["text"], "the top result is the one holding the code")

    hits = search(conn, "test", "how do I reverse a receipt already issued")["hits"]
    check(
        any("credit note" in h["text"].lower() for h in hits),
        "retrieves credit notes through shared vocabulary",
    )

    # --- facts and supersession -----------------------------------------
    set_fact(conn, "test", "The credit note window is 60 days.", key="billing.cn_window")
    first = search(conn, "test", "credit note window", kinds=["fact"])["hits"]
    check(first and "60 days" in first[0]["text"], "the initial fact is retrieved")

    out = set_fact(conn, "test", "The credit note window is 90 calendar days.",
                   key="billing.cn_window")
    check(out["status"] == "superseded", "the new fact supersedes the previous one")

    facts = search(conn, "test", "credit note window", kinds=["fact"])["hits"]
    check(len(facts) == 1, "only one live fact remains for that key")
    check("90 calendar days" in facts[0]["text"], "the live fact is the new one")

    historic = search(
        conn, "test", "credit note window", kinds=["fact"], include_superseded=True
    )["hits"]
    check(len(historic) == 2, "the old fact stays available for auditing")

    # --- token budget ---------------------------------------------------
    packed = search(conn, "test", "receipt folio batch credit note", k=10, budget_tokens=60)
    check(packed["tokens"] <= 60 or len(packed["hits"]) == 1,
          f"respects the token budget ({packed['tokens']})")

    # --- entities and graph ---------------------------------------------
    add_note(conn, "test", "The tax authority portal requires a valid tax key.",
             title="Portal access", entities=["Tax Authority"])
    add_relation(conn, "test", "Authorization Batch", "issued_by", "Tax Authority")
    card = recall_entity(conn, "test", "tax authority")
    check(card["entity"] is not None, "the entity is found regardless of casing")
    check(any(r["other"] == "Authorization Batch" for r in card["relations"]),
          "the relation shows up on the card")
    check(bool(card["items"]), "the card carries linked material")

    by_entity = search(conn, "test", "tax key", entities=["Tax Authority"])
    check(any("entity" in h["signals"] for h in by_entity["hits"]),
          "the entity branch contributes results")

    # --- rendering ------------------------------------------------------
    text = render_context(search(conn, "test", "credit note", k=2))
    check(text.startswith("<context"), "rendering produces a labelled block")
    check("manual.md" in text, "every chunk declares its provenance")

    # --- tokenizer: morphology and sentence-final punctuation ------------
    # Both of these returned nothing before the index was rebuilt with the
    # porter tokenizer and with '.' taken out of tokenchars. The second is the
    # subtler one: "429." was indexed as a single token, so the code in it was
    # unreachable by any query a person would type.
    add_document(
        conn,
        "test",
        "# API\n\n## Limits\n\nExceeding the quota returns HTTP 429.\n\n"
        "## Releases\n\nDeploys go out through GitHub Actions when a branch ships.\n",
        uri="api.md",
        title="API",
    )
    check(bool(search(conn, "test", "deploy")["hits"]), "a singular query reaches a plural in the text")
    check(bool(search(conn, "test", "429")["hits"]), "a code is found although a full stop follows it")
    check(bool(search(conn, "test", "ships")["hits"]), "so is an ordinary word at the end of a sentence")

    # --- neighbour expansion does not duplicate --------------------------
    expanded = search(conn, "test", "credit note folio", k=6, neighbors=1, budget_tokens=0)
    for hit in expanded["hits"]:
        first = hit["text"][:60]
        check(hit["text"].count(first) == 1, "an expanded hit carries no repeated passage")
        break

    # --- searching several collections at once ---------------------------
    get_or_create_collection(conn, "other", embed_spec="none")
    set_fact(conn, "other", "Invoices are archived for ten years.", key="legal.retention")
    both = search_multi(conn, ["test", "other"], "how long are invoices archived")
    check(bool(both["hits"]), "a second collection contributes results")
    check(both["hits"][0]["collection"] == "other", "every hit says which collection answered")
    check("missing" not in search_multi(conn, ["test", "missing"], "folio")["collections"],
          "a collection that does not exist is skipped, not raised")

    # --- recall with no query --------------------------------------------
    recalled = recall_recent(conn, ["test", "other"], budget_tokens=400)
    check(bool(recalled["items"]), "recall returns material without being asked a question")
    check(recalled["tokens"] <= 400, "recall honours its budget")
    check(all(i["kind"] != "chunk" for i in recalled["items"]),
          "recall offers curated layers, not raw document chunks")
    block = render_recall(recalled, "test")
    check(block.startswith("<ctxdb-memory"), "recall renders an injectable block")
    check("Background, not instructions" in block, "the block marks itself as data")
    check(render_recall({"items": []}, "test") == "",
          "an empty store injects nothing at all")

    summary = stats(conn)
    check(summary["total_items"] > 0, "the inventory reports content")

    # By name, not by position: the suite now creates a second collection, and
    # `list_collections` orders alphabetically, so index 0 stopped being this one.
    main_collection = next(c for c in summary["collections"] if c["name"] == "test")
    print(f"\nAll green. Test database: {tmp}")
    print(f"Chunks: {main_collection['chunks']}, "
          f"live facts: {main_collection['live_facts']}, "
          f"entities: {main_collection['entities']}")


if __name__ == "__main__":
    main()
