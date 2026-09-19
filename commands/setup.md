---
description: Build ctxdb's environment so the memory actually runs
allowed-tools: Bash, Read
---

Build the environment this plugin needs, then report honestly whether it worked.

Run it from the plugin root, which is `${CLAUDE_PLUGIN_ROOT}`:

```
uv sync --project "${CLAUDE_PLUGIN_ROOT}" --extra local
```

This downloads a few hundred megabytes the first time — PyTorch and a 120 MB
embedding model's dependencies — and is why it is a deliberate command rather
than something a session-start hook does behind your back. It takes a few
minutes on a cold cache and seconds afterwards.

If the user passes `$ARGUMENTS` containing `no-embeddings`, drop `--extra local`
instead: ctxdb then runs on BM25 alone, with nothing to download. Say plainly
that paraphrased questions will miss in that mode, and that
`ctxdb collection reindex <name> --embeddings local` is how to upgrade later.

Then verify rather than assume:

```
uv run --no-sync --project "${CLAUDE_PLUGIN_ROOT}" ctxdb status
```

Report what it prints — which collection this project writes to, and whether
vectors are active. If the sync failed, show the error rather than summarising
it, and stop.

Finally, tell the user to restart Claude Code, because the MCP server and the
session-start hook are both read at launch and neither will pick this up until
then.
