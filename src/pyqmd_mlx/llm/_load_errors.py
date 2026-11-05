"""One clean message for a model that cannot be loaded or downloaded.

A blocked repository, a TLS-inspecting proxy and being offline all arrive as
different exception types that vary with the huggingface_hub / httpx / requests
versions, so the chain is classified by class name and message, not by import.
The original reason always stays in the message when nothing matches.

See docs/specs/2026-11-04-model-sets-and-config-design.md, section 5.
"""

from ._constants import (
    EXPAND_MODEL_ENV_VAR,
    MODEL_SET_ENV_VAR,
    MODEL_SETS,
    RERANK_MODEL_ENV_VAR,
)

_MAX_REASON = 300

_DENIED = (
    "access to the repository was denied (a network filter may be blocking it, or it needs a login)"
)
_NOT_FOUND = "the repository was not found (check the id; a network filter can also hide it)"
_TLS = (
    "the connection failed TLS verification (behind a proxy that inspects traffic, point "
    "SSL_CERT_FILE at your organisation's CA certificate bundle)"
)
_UNREACHABLE = "Hugging Face could not be reached (offline, or blocked by a firewall)"

_TLS_NAMES = ("ssl", "certificate")
_TLS_TEXT = ("certificate_verify_failed", "certificate verify failed", "ssl:")
_CONNECTION_NAMES = (
    "connecterror",
    "connectionerror",
    "connecttimeout",
    "readtimeout",
    "timeout",
    "offlinemodeisenabled",
    "localentrynotfounderror",
    "newconnectionerror",
    "maxretryerror",
)
_CONNECTION_TEXT = (
    "name or service not known",
    "nodename nor servname",
    "temporary failure in name resolution",
    "network is unreachable",
    "connection refused",
    "connection reset",
    "connection aborted",
    "outgoing traffic has been disabled",
)


def _chain(exc: BaseException) -> list[BaseException]:
    seen: list[BaseException] = []
    while exc is not None and exc not in seen and len(seen) < 10:
        seen.append(exc)
        exc = exc.__cause__ or exc.__context__
    return seen


def _names(exc: BaseException) -> list[str]:
    return [cls.__name__.lower() for cls in type(exc).__mro__]


def _status(exc: BaseException) -> int | None:
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _one_line(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()[0].strip() if str(exc).strip() else type(exc).__name__
    text = text.rstrip(".")
    return text if len(text) <= _MAX_REASON else text[: _MAX_REASON - 3] + "..."


def explain_failure(exc: BaseException) -> str:
    """A short reason for a failed model load or download."""
    chain = _chain(exc)
    texts = [str(e).lower() for e in chain]
    if any(any(t in name for t in _TLS_NAMES) for e in chain for name in _names(e)) or any(
        t in text for text in texts for t in _TLS_TEXT
    ):
        return _TLS
    for e in chain:
        names = _names(e)
        status = _status(e)
        if "gatedrepoerror" in names or status in (401, 403):
            return _DENIED
        if "repositorynotfounderror" in names or status == 404:
            return _NOT_FOUND
    if any(n in _CONNECTION_NAMES for e in chain for n in _names(e)) or any(
        t in text for text in texts for t in _CONNECTION_TEXT
    ):
        return _UNREACHABLE
    return _one_line(exc)


def _default_model(role: str) -> str | None:
    default = MODEL_SETS["default"]
    return {"expand": default.expand, "rerank": default.rerank}.get(role)


def load_failure_message(
    label: str,
    model_id: str,
    exc: BaseException,
    *,
    role: str,
    env_var: str | None = None,
    verb: str = "load",
) -> str:
    """`Could not <verb> <label> model '<id>': <reason>. <remedy>`.

    The remedy for a built-in default-set model also offers the Granite set;
    for an explicit choice it names the variable to change. The embedding
    model has no alternative, so its remedy is about reaching the Hub."""
    reason = explain_failure(exc)
    head = f"Could not {verb} {label} model '{model_id}': {reason}."
    if role == "embed":
        return (
            f"{head} Check that huggingface.co is reachable from this machine, or run "
            "'pyqmd pull' once on a network that can reach it to cache the model."
        )
    override = (
        f"Set {env_var} to a Hugging Face repo id or a local MLX model directory to override."
    )
    if model_id == _default_model(role):
        return (
            f"{head} To use the IBM Granite models instead, run 'pyqmd models use granite' "
            f"(or set {MODEL_SET_ENV_VAR}=granite). {override}"
        )
    return f"{head} {override}"


ROLE_ENV_VARS = {"expand": EXPAND_MODEL_ENV_VAR, "rerank": RERANK_MODEL_ENV_VAR}
