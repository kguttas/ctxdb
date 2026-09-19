---
description: Show what ctxdb has stored for this project
allowed-tools: mcp__ctxdb__context_status, mcp__ctxdb__context_search, mcp__ctxdb__context_recall_entity
---

Report what is stored: $ARGUMENTS

With no arguments, call `context_status` and summarise what each collection holds
and which one this project writes to. With arguments, `context_search` them across
the project and global scopes and show what comes back, with its provenance.

Report only what the tools return. If nothing is stored, say so plainly.
