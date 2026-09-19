#!/bin/sh
# Hands the session whatever is stored for its project, plus the rule for what is
# worth storing.
#
# The interesting case is the one where it cannot: on a fresh install nothing has
# been built yet, and a memory plugin that fails quietly is the exact failure this
# project exists to remove. So a missing environment says so, once, instead of
# printing nothing and leaving everything looking fine.
set -u

ROOT="${CLAUDE_PLUGIN_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"

if [ ! -x "$ROOT/.venv/bin/python" ] && [ ! -f "$ROOT/.venv/Scripts/python.exe" ]; then
    printf '%s' '{"systemMessage":"ctxdb is installed but has no environment yet, so nothing is being remembered. Run /ctxdb:setup once to build it."}'
    exit 0
fi

# --no-sync is not an optimisation, it is a requirement: a plain `uv run`
# re-syncs the environment on every invocation, and on Windows that collides
# with the DLLs the running MCP server holds open — which corrupts the very
# environment this hook depends on.
exec uv run --no-sync --project "$ROOT" ctxdb recall --hook --policy 2>/dev/null
