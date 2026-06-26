from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any, Callable


@dataclass(frozen=True)
class NoteRetrievalResult:
    file_path: str
    chunk_id: str
    snippet: str
    content: str
    score: float


MarkdownFiles = Callable[[Path], list[Path]]
RelativePath = Callable[[Path, Path], str]


def retrieve_notes_from_vault(
    vault_path: str | Path,
    query: str,
    max_results: int = 5,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    markdown_files: MarkdownFiles | None = None,
    relative_path: RelativePath | None = None,
) -> list[dict[str, Any]]:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    vault = Path(vault_path).expanduser().resolve()
    chunks = _build_note_chunks(
        vault,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        markdown_files=markdown_files or _default_markdown_files,
        relative_path=relative_path or _default_relative_path,
    )
    if not chunks:
        return []

    document_frequency: dict[str, int] = {}
    chunk_tokens: list[list[str]] = []
    for chunk in chunks:
        tokens = _tokenize(f"{chunk['file_path']} {chunk['content']}")
        chunk_tokens.append(tokens)
        for token in set(tokens):
            document_frequency[token] = document_frequency.get(token, 0) + 1

    query_weights = _tfidf_weights(query_tokens, document_frequency, len(chunks))
    scored_results: list[NoteRetrievalResult] = []

    for chunk, tokens in zip(chunks, chunk_tokens):
        weights = _tfidf_weights(tokens, document_frequency, len(chunks))
        score = _cosine_similarity(query_weights, weights)
        if score <= 0:
            continue
        scored_results.append(
            NoteRetrievalResult(
                file_path=chunk["file_path"],
                chunk_id=chunk["chunk_id"],
                snippet=_build_retrieval_snippet(chunk["content"], query_tokens),
                content=chunk["content"],
                score=round(score, 4),
            )
        )

    scored_results.sort(key=lambda item: item.score, reverse=True)
    return [result.__dict__ for result in scored_results[:max_results]]


def _default_markdown_files(vault: Path) -> list[Path]:
    if not vault.exists():
        return []
    return sorted(path for path in vault.rglob("*.md") if path.is_file())


def _default_relative_path(path: Path, vault: Path) -> str:
    return path.relative_to(vault).as_posix()


def _read_markdown(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def _build_note_chunks(
    vault: Path,
    chunk_size: int,
    chunk_overlap: int,
    markdown_files: MarkdownFiles,
    relative_path: RelativePath,
) -> list[dict[str, str]]:
    chunks: list[dict[str, str]] = []
    overlap = min(chunk_overlap, chunk_size - 1)

    for note_path in markdown_files(vault):
        note_relative_path = relative_path(note_path, vault)
        content = _read_markdown(note_path)
        if not content.strip():
            continue

        start = 0
        index = 0
        while start < len(content):
            end = min(start + chunk_size, len(content))
            chunk_text = content[start:end].strip()
            if chunk_text:
                chunks.append(
                    {
                        "file_path": note_relative_path,
                        "chunk_id": f"{note_relative_path}#{index}",
                        "content": chunk_text,
                    }
                )
            if end == len(content):
                break
            start = max(end - overlap, start + 1)
            index += 1

    return chunks


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text.casefold())


def _tfidf_weights(tokens: list[str], document_frequency: dict[str, int], document_count: int):
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1

    total = max(len(tokens), 1)
    weights = {}
    for token, count in counts.items():
        idf = math.log((1 + document_count) / (1 + document_frequency.get(token, 0))) + 1
        weights[token] = (count / total) * idf
    return weights


def _cosine_similarity(left, right) -> float:
    shared = set(left) & set(right)
    if not shared:
        return 0

    numerator = sum(left[token] * right[token] for token in shared)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0
    return numerator / (left_norm * right_norm)


def _build_retrieval_snippet(content: str, query_tokens: list[str], radius: int = 90) -> str:
    folded = content.casefold()
    positions = [folded.find(token) for token in query_tokens if folded.find(token) >= 0]
    if not positions:
        return content[: radius * 2].replace("\r", " ").replace("\n", " ").strip()

    first = min(positions)
    start = max(first - radius, 0)
    end = min(first + radius, len(content))
    snippet = content[start:end].replace("\r", " ").replace("\n", " ").strip()
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(content) else ""
    return f"{prefix}{snippet}{suffix}"
