"""The rule for what is worth remembering.

It lives in the package, not in a CLAUDE.md, because it has to travel with the
plugin. A user who installs ctxdb gets the tools, the recall hook and this in one
piece; asking them to also paste a policy into their own configuration is asking
them to do the step that, left undone, is the whole failure this project exists
to fix.

It is injected through the same SessionStart hook that carries recalled memory.
That costs its own tokens in every session, which is the honest price: an
instruction that is only sometimes in context is an instruction that is only
sometimes followed.
"""

from __future__ import annotations

MEMORY_POLICY = """\
<ctxdb-policy>
The context_* tools are a memory that outlives this session.

READING. Search before answering from assumption whenever the question touches a
past decision, a convention, or a detail you would otherwise be guessing at.
context_search covers this project and the global memory at once. It is cheap;
answering confidently from a stale memory is not. If a search returns nothing,
context_status says whether that means "never stored" or "stored elsewhere".

WRITING. Store what is durable, non-obvious, and not already written somewhere
the next session will look:
  - a decision AND its reason (the code shows the decision, never the reason);
  - a correction the user made to how you work — the most valuable, most lost;
  - a constraint that cost time to find: an environment quirk, why the obvious
    approach fails here;
  - a location that took real effort to locate.

Do not store what the repository, git history or a CLAUDE.md already records,
the narrative of what you just did, secrets or credentials, or anything that
dies with this conversation. A handful of entries per session is normal. When
unsure, store nothing: a store full of noise is how retrieval stops being worth
reading.

WHICH TOOL. context_set_fact for anything that changes — decisions, preferences,
state, versions, owners — always with a stable dotted key (arch.database,
pref.answer_style). That key is what makes a new value supersede the old one
instead of both surfacing later and contradicting each other. context_add_note
for a standing observation with no natural key. context_add_document for long
reference text, with a stable uri.

WHICH SCOPE. Leave collection empty to file it under this project. Pass
collection="global" for what is true of the user everywhere: how they like
answers written, tools they always reach for, standing instructions. A
preference stored per-project has to be relearned in every other one.

Correct a stored fact by writing under its existing key, not by adding a second
entry. Use context_forget when something was wrong rather than merely outdated.
</ctxdb-policy>"""
