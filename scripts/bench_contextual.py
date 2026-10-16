"""Does an LLM-written context line per chunk beat heading breadcrumbs?

Two subcommands, both resumable and meant to be run by hand:
  generate --generator small|mid   write a context line for every chunk
  score                            embed base / crumbs / ctx-small / ctx-mid and
                                   apply the rule fixed in the spec
The rule and every choice here are fixed in advance in the spec
(2026-10-16-contextual-embedding-experiment-design.md). Never touches the
user's index.
"""

import hashlib
import json
import re
from pathlib import Path

GENERATORS = {
    "small": "mlx-community/Qwen3-1.7B-4bit",
    "mid": "mlx-community/Qwen3-4B-4bit",
}
MAX_WINDOW_CHARS = 18_000
MAX_NEW_TOKENS = 60

_INSTRUCTION = (
    "Please give a short succinct context to situate this chunk within the overall "
    "document for the purpose of improving search retrieval of the chunk. "
    "Answer only with the succinct context and nothing else."
)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def window_text(page: str, pos: int, length: int, max_chars: int = MAX_WINDOW_CHARS) -> str:
    """The page, or max_chars of it centred on the chunk [pos, pos + length)."""
    if len(page) <= max_chars:
        return page
    centre = pos + length // 2
    start = min(max(centre - max_chars // 2, 0), len(page) - max_chars)
    return page[start : start + max_chars]


def build_prompt(window: str, chunk: str) -> str:
    return (
        f"<document>\n{window}\n</document>\n"
        "Here is the chunk we want to situate within the whole document\n"
        f"<chunk>\n{chunk}\n</chunk>\n"
        f"{_INSTRUCTION}"
    )


def clean_context(raw: str) -> str:
    """One line: reasoning blocks (closed, or open to the end) removed, whitespace collapsed."""
    text = _THINK_BLOCK.sub("", raw)
    text = text.split("<think>", 1)[0]
    return " ".join(text.split())


def ctx_text(context: str, chunk: str) -> str:
    return f"{context}\n\n{chunk}" if context else chunk


def context_key(generator_id: str, prompt: str) -> str:
    """Covers the generator, the instruction, the page window and the chunk
    (the prompt contains all of them)."""
    return hashlib.sha256(f"{generator_id}\x00{prompt}".encode()).hexdigest()


class ContextCache:
    """key -> context line, appended to a JSONL file as it is generated. A line
    torn by a kill is dropped on load."""

    def __init__(self, path: Path | None):
        self.path = path
        self._items: dict[str, str] = {}
        if path is not None and path.exists():
            good = []
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._items[record["key"]] = record["context"]
                good.append(line)
            path.write_text("".join(f"{line}\n" for line in good), encoding="utf-8")

    def get(self, key: str) -> str | None:
        return self._items.get(key)

    def put(self, key: str, context: str) -> None:
        self._items[key] = context
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps({"key": key, "context": context}, ensure_ascii=False)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
