"""Download the models pyqmd would use, without loading them.

`pyqmd pull` and `pyqmd models use --download` share this. Which ids to fetch
comes from the caller (the resolved roles, or one named model set), so a model
the user did not select is never requested.

See docs/specs/2026-11-04-model-sets-and-config-design.md.
"""

from dataclasses import dataclass
from pathlib import Path

from ._constants import DEFAULT_EMBED_MODEL
from ._hub import is_downloaded
from ._load_errors import ROLE_ENV_VARS, load_failure_message
from ._models import resolve_model

# role -> the word used in messages
LABELS = {"embed": "embedding", "rerank": "rerank", "expand": "query-expansion"}


@dataclass(frozen=True)
class DownloadResult:
    role: str
    model_id: str
    status: str  # "downloaded" | "cached" | "local" | "failed"
    message: str = ""  # the full error line when status is "failed"


def active_models() -> list[tuple[str, str]]:
    """(role, model id) for the embedding, rerank and expansion models a query
    would use right now."""
    return [
        ("embed", DEFAULT_EMBED_MODEL),
        ("rerank", resolve_model("rerank")),
        ("expand", resolve_model("expand")),
    ]


def download_models(models: list[tuple[str, str]]) -> list[DownloadResult]:
    """Fetch each (role, id) that is not already local or cached. A failure is
    reported in its result; the remaining models are still attempted."""
    import huggingface_hub

    results = []
    for role, model_id in models:
        if Path(model_id).is_dir():
            results.append(DownloadResult(role, model_id, "local"))
        elif is_downloaded(model_id):
            results.append(DownloadResult(role, model_id, "cached"))
        else:
            try:
                huggingface_hub.snapshot_download(model_id)
            except Exception as exc:
                message = load_failure_message(
                    LABELS[role],
                    model_id,
                    exc,
                    role=role,
                    env_var=ROLE_ENV_VARS.get(role),
                    verb="download",
                )
                results.append(DownloadResult(role, model_id, "failed", message))
            else:
                results.append(DownloadResult(role, model_id, "downloaded"))
    return results
