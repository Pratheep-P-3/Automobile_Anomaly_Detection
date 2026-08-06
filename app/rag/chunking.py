"""Manual chunking utilities for service manuals."""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    heading: str
    text: str


def chunk_markdown(text: str, source_id: str) -> list[Chunk]:
    """Split a markdown document into chunks along '## ' section headers.

    Each returned chunk includes the document title (H1) as context so that
    embeddings retain top-level topic information even for small sections.
    """
    lines = text.splitlines()
    title = ""
    for line in lines:
        if line.startswith("# "):
            title = line[2:].strip()
            break

    # Split on level-2 headings while keeping the heading text.
    sections = re.split(r"(?m)^## ", text)
    chunks: list[Chunk] = []
    idx = 0
    for section in sections:
        section = section.strip()
        if not section or section.startswith("# "):
            continue
        heading_line, *rest = section.split("\n", 1)
        heading = heading_line.strip()
        body = rest[0].strip() if rest else ""
        combined = f"{title} - {heading}\n{body}".strip()
        if combined:
            idx += 1
            chunks.append(Chunk(chunk_id=f"{source_id}::chunk{idx}", heading=heading, text=combined))
    return chunks


def chunk_manual_text(text: str, source_id: str) -> list[Chunk]:
    """Chunk extracted manual text from PDFs or markdown-like sources."""
    markdown_chunks = chunk_markdown(text, source_id)
    if markdown_chunks:
        return markdown_chunks

    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
    chunks: list[Chunk] = []
    for idx, paragraph in enumerate(paragraphs, start=1):
        heading = paragraph.splitlines()[0][:80]
        chunks.append(Chunk(chunk_id=f"{source_id}::chunk{idx}", heading=heading, text=paragraph))
    return chunks
