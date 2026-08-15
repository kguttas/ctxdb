"""Structure-aware chunking.

The quality of a system like this is decided here, not in the database. Cutting
blindly every 500 tokens splits tables in half, separates a claim from the
condition that qualifies it, and leaves fragments that mean nothing on their own.

Three rules drive this splitter:

1. Headings win. A chunk never crosses a heading, and it carries its full
   heading path ("Manual > Billing > Credit notes") so it keeps its context when
   retrieved in isolation.
2. Code blocks and tables are atomic: they are split only if they exceed the
   maximum all by themselves.
3. Overlap is measured in whole paragraphs, not characters, so no sentence ever
   ends up cut in half.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .db import estimate_tokens

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
SENTENCE_RE = re.compile(r"(?<=[.!?¿?])\s+(?=[A-ZÁÉÍÓÚÑ¿¡])")


@dataclass
class Chunk:
    text: str
    heading_path: str = ""
    ord: int = 0


@dataclass
class _Section:
    heading_path: str
    blocks: list[str] = field(default_factory=list)


def split_sections(text: str) -> list[_Section]:
    """Split on markdown headings, ignoring any that appear inside code fences."""
    sections: list[_Section] = []
    stack: list[str] = []
    current = _Section(heading_path="")
    buffer: list[str] = []
    in_fence = False

    def flush_block() -> None:
        block = "\n".join(buffer).strip()
        if block:
            current.blocks.append(block)
        buffer.clear()

    for line in text.splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
            buffer.append(line)
            continue

        heading = None if in_fence else HEADING_RE.match(line)
        if heading:
            flush_block()
            if current.blocks:
                sections.append(current)
            level, title = len(heading.group(1)), heading.group(2)
            stack = stack[: level - 1]
            while len(stack) < level - 1:
                stack.append("")
            stack.append(title)
            current = _Section(heading_path=" > ".join(p for p in stack if p))
            continue

        if not in_fence and not line.strip():
            flush_block()
        else:
            buffer.append(line)

    flush_block()
    if current.blocks:
        sections.append(current)
    return sections


def _split_oversized(block: str, max_tokens: int) -> list[str]:
    """Last resort: a single block that already exceeds the maximum by itself."""
    pieces = SENTENCE_RE.split(block) if not FENCE_RE.match(block) else block.splitlines()
    out: list[str] = []
    buf: list[str] = []
    size = 0
    for piece in pieces:
        n = estimate_tokens(piece)
        if buf and size + n > max_tokens:
            out.append("\n".join(buf) if "\n" in block else " ".join(buf))
            buf, size = [], 0
        buf.append(piece)
        size += n
    if buf:
        out.append("\n".join(buf) if "\n" in block else " ".join(buf))
    return [p.strip() for p in out if p.strip()]


def chunk_text(
    text: str,
    target_tokens: int = 350,
    max_tokens: int = 600,
    overlap_blocks: int = 1,
) -> list[Chunk]:
    """Turn a document into retrievable chunks.

    `target_tokens` is the size at which a chunk is closed; `max_tokens` is the
    hard ceiling. `overlap_blocks` repeats the last N paragraphs at the start of
    the next chunk so the thread is not lost between them.
    """
    chunks: list[Chunk] = []
    position = 0

    for section in split_sections(text):
        buf: list[str] = []
        size = 0

        def close() -> None:
            nonlocal buf, size, position
            body = "\n\n".join(buf).strip()
            if body:
                chunks.append(Chunk(text=body, heading_path=section.heading_path, ord=position))
                position += 1
            buf, size = [], 0

        for block in section.blocks:
            n = estimate_tokens(block)

            if n > max_tokens:
                close()
                for piece in _split_oversized(block, max_tokens):
                    chunks.append(
                        Chunk(text=piece, heading_path=section.heading_path, ord=position)
                    )
                    position += 1
                continue

            if buf and size + n > target_tokens:
                tail = buf[-overlap_blocks:] if overlap_blocks else []
                close()
                # Overlap is only worth it if it still leaves room for new content.
                if tail and sum(estimate_tokens(t) for t in tail) + n <= max_tokens:
                    buf = list(tail)
                    size = sum(estimate_tokens(t) for t in tail)

            buf.append(block)
            size += n

        close()

    return chunks
