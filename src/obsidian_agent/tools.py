from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_core.tools import tool


@dataclass(frozen=True)
class NoteSearchResult:
    file_path: str
    snippet: str


def _resolve_vault(vault_path: str | Path) -> Path:
    return Path(vault_path).expanduser().resolve()


def _markdown_files(vault_path: str | Path) -> list[Path]:
    vault = _resolve_vault(vault_path)
    if not vault.exists():
        return []
    return sorted(path for path in vault.rglob("*.md") if path.is_file())


def _relative_posix(path: Path, vault: Path) -> str:
    return path.relative_to(vault).as_posix()


def _safe_note_path(vault_path: str | Path, file_path: str | Path) -> Path:
    vault = _resolve_vault(vault_path)
    candidate = Path(file_path)
    if not candidate.is_absolute():
        candidate = vault / candidate

    resolved = candidate.resolve()
    if not resolved.is_relative_to(vault):
        raise ValueError("Refusing to read a file outside the configured Obsidian vault.")
    if resolved.suffix.lower() != ".md":
        raise ValueError("Only Markdown notes can be read.")
    return resolved


def search_notes_in_vault(
    vault_path: str | Path,
    query: str,
    max_results: int = 5,
) -> list[dict[str, str]]:
    terms = [term.casefold() for term in query.split() if term.strip()]
    if not terms:
        return []

    vault = _resolve_vault(vault_path)
    results: list[NoteSearchResult] = []

    for note_path in _markdown_files(vault):
        relative_path = _relative_posix(note_path, vault)
        try:
            content = note_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = note_path.read_text(encoding="utf-8", errors="ignore")

        haystack = f"{relative_path}\n{content}".casefold()
        if not all(term in haystack for term in terms):
            continue

        snippet = _build_snippet(relative_path, content, terms)
        results.append(NoteSearchResult(file_path=relative_path, snippet=snippet))
        if len(results) >= max_results:
            break

    return [result.__dict__ for result in results]


def read_note_from_vault(vault_path: str | Path, file_path: str | Path) -> dict[str, str]:
    vault = _resolve_vault(vault_path)
    note_path = _safe_note_path(vault, file_path)
    content = note_path.read_text(encoding="utf-8")
    return {"file_path": _relative_posix(note_path, vault), "content": content}


def list_notes_in_vault(vault_path: str | Path) -> list[str]:
    vault = _resolve_vault(vault_path)
    return [_relative_posix(path, vault) for path in _markdown_files(vault)]


def create_note_tools(vault_path: str | Path, max_results: int = 5):
    @tool
    def search_notes(query: str) -> list[dict[str, str]]:
        """Search Markdown notes by local keyword matching. Returns file_path and snippet."""
        return search_notes_in_vault(vault_path, query, max_results=max_results)

    @tool
    def read_note(file_path: str) -> dict[str, str]:
        """Read one Markdown note from inside the configured Obsidian vault."""
        return read_note_from_vault(vault_path, file_path)

    @tool
    def list_notes() -> list[str]:
        """List Markdown notes in the configured Obsidian vault. Useful for debugging."""
        return list_notes_in_vault(vault_path)

    return [search_notes, read_note, list_notes]


def _build_snippet(relative_path: str, content: str, terms: list[str], radius: int = 90) -> str:
    folded = content.casefold()
    positions = [folded.find(term) for term in terms if folded.find(term) >= 0]
    if not positions:
        return relative_path

    first = min(positions)
    start = max(first - radius, 0)
    end = min(first + radius, len(content))
    snippet = content[start:end].replace("\r", " ").replace("\n", " ").strip()
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(content) else ""
    return f"{prefix}{snippet}{suffix}"
