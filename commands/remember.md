---
description: Store something in ctxdb so it survives this session
---

Store what the user asked to remember: $ARGUMENTS

If `$ARGUMENTS` is empty, look back over this conversation and pick what is worth
keeping by the bar in CLAUDE.md — durable, non-obvious, not already recorded in
the repository. Show the candidates and ask before writing more than two.

Use `context_set_fact` with a stable dotted key for anything that can change, and
`collection="global"` when it is true of the user everywhere rather than of this
project. Before writing, run `context_search` on the same topic: if a fact already
covers it, update it under its existing key instead of creating a near-duplicate.

Then report in one line what was stored, under which key and in which scope.
