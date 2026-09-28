"""Model-call wrappers for bench --samples (see
docs/specs/2026-09-25-bench-samples-design.md). Installed on a
Store with Store.wrapping_llm_fns, so bench still measures the real
Store.query pipeline."""

import hashlib


def salted_memo(expand_fn, salt: str):
    """expand_fn with a fixed salt, memoized by (query, model): within one
    sample, hybrid and full share a single draw per query. Failures aren't
    memoized, so each backend sees the error the way it would unwrapped."""
    memo: dict[tuple[str, str], list[str]] = {}

    def expand(query: str, model: str) -> list[str]:
        key = (query, model)
        if key not in memo:
            memo[key] = expand_fn(query, model, salt=salt)
        return memo[key]

    return expand


class RerankCache:
    """Rerank scores by (query, doc) across one bench run. Reranking is a
    pure function of the pair, and samples share most of their candidates.

    Sample 0 uses recording(): every call still reaches the reranker with
    its full batch, so sample 0 matches a plain run. Later samples use
    caching(), which sends only unseen docs. A cached score can differ in
    the last bits from one computed in a different batch -- the replay
    harness accepts the same."""

    def __init__(self) -> None:
        self._scores: dict[tuple[str, str], float] = {}

    @staticmethod
    def _key(query: str, doc: str) -> tuple[str, str]:
        return (query, hashlib.sha1(doc.encode("utf-8")).hexdigest())

    def recording(self, rerank_fn):
        def rerank(query: str, documents: list[str], model: str) -> list[float]:
            scores = rerank_fn(query, documents, model)
            for doc, score in zip(documents, scores):
                self._scores[self._key(query, doc)] = score
            return scores

        return rerank

    def caching(self, rerank_fn):
        def rerank(query: str, documents: list[str], model: str) -> list[float]:
            keys = [self._key(query, doc) for doc in documents]
            unseen: dict[tuple[str, str], str] = {}
            for key, doc in zip(keys, documents):
                if key not in self._scores:
                    unseen.setdefault(key, doc)
            if unseen:
                scores = rerank_fn(query, list(unseen.values()), model)
                self._scores.update(zip(unseen.keys(), scores))
            return [self._scores[key] for key in keys]

        return rerank
