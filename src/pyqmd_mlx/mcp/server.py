"""MCP server factory and transport entry points, shared by stdio (Task 14)
and Streamable HTTP (Task 15). Registers the qmd://{+path} resource (Task
13) and the query/get/multi_get/status tools (Tasks 9-12) onto one
MCPServer instance built from a live Store.

Concurrency: the official mcp SDK dispatches sync tool functions through
anyio.to_thread.run_sync -- a real thread pool (verified during design
against the actual installed SDK source: mcp/server/mcpserver/utilities/
func_metadata.py's call_fn). Two concurrent MCP tool calls can run their
handler bodies on two different worker threads at the same time. Python's
sqlite3.Connection is not safe for concurrent multi-threaded use of one
connection object -- unlike the Node reference, whose single-threaded event
loop already serializes all SQL execution for free, no matter how many
requests appear concurrent. `_locked` wraps every Store call a tool handler
makes in one process-wide lock, serializing access: the closer behavioral
match to how the Node version already works today, not a new restriction
relative to parity. The lock's scope is intentionally whatever `Store.query()`
does internally, not just SQL: for the `query` tool that includes query
expansion, embedding, and reranking (real LLM inference, potentially seconds
of wall-clock time), so a `query` call serializes other requests for its
full duration -- again matching the Node reference's de facto
single-threaded behavior, not a new restriction.
"""

import signal
import sys
import threading
import time
from collections.abc import Callable
from typing import TypeVar
from urllib.parse import quote, urlparse

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ResourceNotFoundError
from mcp.types import CallToolResult, EmbeddedResource, TextContent, TextResourceContents
from pydantic import StrictInt

from pyqmd_mlx.cli._lineparse import parse_line_range
from pyqmd_mlx.cli._multiget import resolve_multi_get
from pyqmd_mlx.mcp._errors import tool_result_or_error
from pyqmd_mlx.mcp._formatting import format_query_result, format_query_summary
from pyqmd_mlx.mcp._instructions import build_instructions
from pyqmd_mlx.store import Store

T = TypeVar("T")

_lock = threading.Lock()


def _locked(fn: Callable[[], T]) -> T:
    with _lock:
        return fn()


def _package_version() -> str:
    from pyqmd_mlx.version import __version__

    return __version__


def _status_impl(store: Store) -> CallToolResult:
    def _run() -> CallToolResult:
        counts = _locked(lambda: store.get_status_counts())
        collections = _locked(lambda: store.list_collections())

        summary_lines = [
            "qmd Index Status:",
            f"  Active documents: {counts['active_documents']}",
            f"  Embedded vectors: {counts['embedded_vectors']}",
            f"  Pending embed: {counts['pending_embed']}",
            f"  Collections: {len(collections)}",
        ]
        for c in collections:
            summary_lines.append(f"    - {c['name']}: {c['path']}")

        return CallToolResult(
            content=[TextContent(type="text", text="\n".join(summary_lines))],
            structured_content={"counts": counts, "collections": collections},
        )

    return tool_result_or_error(_run)


def _get_impl(
    store: Store,
    file: str,
    from_line: int | None,
    max_lines: int | None,
    line_numbers: bool,
) -> CallToolResult:
    def _run() -> CallToolResult:
        bare_identifier, resolved_from, resolved_max = parse_line_range(file, from_line, max_lines)
        doc = _locked(lambda: store.find_document_by_identifier(bare_identifier))
        if doc is None:
            # Tier 2 deferred: no Store candidate API for Node's
            # `similarFiles` "Did you mean" block -- grep of
            # src/pyqmd_mlx/store/ shows no similar-file lookup (only an
            # unrelated vector-similarity comment in store.py), so there is
            # nothing cheap to wire up and no retrieval is invented here.
            raise ValueError(f"Document not found: {file}")

        body = doc["doc"]
        start_line = resolved_from or 1
        if resolved_from is not None or resolved_max is not None:
            lines = body.split("\n")
            start = start_line - 1
            end = start + resolved_max if resolved_max is not None else len(lines)
            body = "\n".join(lines[start:end])

        if line_numbers:
            from pyqmd_mlx.cli._output_search import add_line_numbers

            body = add_line_numbers(body, start_line)

        # Node parity (server.ts): `<!-- Context: ... -->` header prefix
        # when the document has collection context. The doc row itself
        # carries no context column -- context lives on the collections
        # table -- so this reads the existing Store API the CLI `get`
        # command already uses, rather than the sketch's `doc.get(...)`
        # (which could never fire).
        context = _locked(lambda: store.get_context_for_path(doc["collection"], doc["path"]))
        if context:
            body = f"<!-- Context: {context} -->\n\n" + body

        display_path = f"{doc['collection']}/{doc['path']}"
        # Node parity: URI-segment encoding (encodeQmdPath) -- encode each
        # path segment separately so slashes survive. encodeURIComponent
        # leaves !*'() unescaped, so they stay safe here too.
        uri = "qmd://" + "/".join(quote(seg, safe="!*'()") for seg in display_path.split("/"))
        return CallToolResult(
            content=[
                EmbeddedResource(
                    type="resource",
                    resource=TextResourceContents(
                        uri=uri,
                        mime_type="text/markdown",
                        text=body,
                        # Node's `get` resource carries `name`/`title` as
                        # top-level sibling fields on the resource object --
                        # not part of the MCP spec's TextResourceContents
                        # schema (uri/mimeType/text/_meta only), confirmed
                        # against a live Node MCP server. The SDK's typed
                        # CallToolResult/EmbeddedResource construction
                        # strips genuinely unknown fields during
                        # serialization, so there's no clean way to emit
                        # them as top-level siblings like Node does;
                        # `_meta` is the spec's actual blessed extension
                        # point for exactly this kind of extra data.
                        meta={"name": display_path, "title": doc["title"]},
                    ),
                )
            ]
        )

    return tool_result_or_error(_run)


def _multi_get_impl(
    store: Store,
    pattern: str,
    max_lines: int | None,
    max_bytes: int = 65536,
    line_numbers: bool = True,
) -> CallToolResult:
    def _run() -> CallToolResult:
        # Node multiGet: `maxBytes || DEFAULT_MULTI_GET_MAX_BYTES` -- an
        # explicit 0 (or null) falls back to the default instead of sizing
        # every file out.
        effective_max_bytes = max_bytes or 65536
        entries = _locked(lambda: resolve_multi_get(store, pattern, effective_max_bytes))
        if not entries or all(entry.not_found for entry in entries):
            raise ValueError(f"No files matched pattern: {pattern}")

        from pyqmd_mlx.cli._output_search import add_line_numbers

        content = []
        for entry in entries:
            if entry.not_found:
                content.append(TextContent(type="text", text=f"Not found: {entry.not_found}"))
                continue
            if entry.skipped:
                # Node shows the bare path here, and the reason comes from
                # the entry's structured skip_detail -- never string-surgery
                # on the CLI-owned skip_reason sentence.
                bare = entry.display_path.removeprefix("qmd://")
                reason = entry.skip_detail or entry.skip_reason or "skipped"
                content.append(
                    TextContent(
                        type="text",
                        text=f"[SKIPPED: {bare} - {reason}. Use 'qmd_get' with file=\"{bare}\" to retrieve.]",
                    )
                )
                continue

            body = entry.body
            if max_lines is not None:
                lines = body.split("\n")
                truncated = len(lines) > max_lines
                body = "\n".join(lines[:max_lines])
                if truncated:
                    body += f"\n\n[... truncated {len(lines) - max_lines} more lines]"
            if line_numbers:
                body = add_line_numbers(body)

            # Node parity (server.ts): per-entry `<!-- Context: ... -->`
            # header + URI-segment encoding. MultiGetEntry carries neither
            # collection/path fields nor context, so both derive from the
            # qmd://-prefixed display_path via the existing
            # get_context_for_path Store API (same source the CLI uses).
            bare = entry.display_path.removeprefix("qmd://")
            collection, _, path = bare.partition("/")
            context = _locked(lambda: store.get_context_for_path(collection, path))
            if context:
                body = f"<!-- Context: {context} -->\n\n" + body
            uri = "qmd://" + "/".join(quote(seg, safe="!*'()") for seg in bare.split("/"))

            content.append(
                EmbeddedResource(
                    type="resource",
                    resource=TextResourceContents(
                        # entry.display_path is now already qmd://-prefixed
                        # (see pyqmd_mlx/cli/_multiget.py) -- prepending it again
                        # here would double it.
                        uri=uri,
                        mime_type="text/markdown",
                        text=body,
                        # See the matching comment in _get_impl: Node's
                        # `multi_get` resources also carry `name`/`title` --
                        # `name` is the bare path there, unlike `uri`.
                        meta={
                            "name": bare,
                            "title": entry.title,
                        },
                    ),
                )
            )
        return CallToolResult(content=content)

    return tool_result_or_error(_run)


def _query_impl(
    store: Store,
    query: str,
    limit: int,
    min_score: float,
    collections: list[str] | None,
    intent: str | None,
    rerank: bool,
    filter: dict | None = None,
    candidate_limit: int | None = None,
) -> CallToolResult:
    from pyqmd_mlx.mcp._errors import error_result
    from pyqmd_mlx.store._metadata_filter import MetadataFilterError, parse_metadata_filter

    parsed_filter = None
    if filter is not None:
        try:
            parsed_filter = parse_metadata_filter(filter)
        except MetadataFilterError as exc:
            return error_result(f"Error: {exc}")

    def _run() -> CallToolResult:
        extra: dict = {}
        if candidate_limit is not None:
            # None means "Store default" (RERANK_CANDIDATE_LIMIT); only
            # override when the caller said a number.
            extra["candidate_limit"] = candidate_limit
        raw_results = _locked(
            lambda: store.query(
                query,
                limit=limit,
                min_score=min_score,
                collection=collections,
                skip_rerank=not rerank,
                intent=intent,
                filter=parsed_filter,
                **extra,
            )
        )
        formatted = [format_query_result(r, query, intent) for r in raw_results]
        return CallToolResult(
            content=[TextContent(type="text", text=format_query_summary(formatted, query))],
            structured_content={"results": formatted},
        )

    return tool_result_or_error(_run)


def _read_document_impl(store: Store, path: str) -> str:
    from pyqmd_mlx.cli._output_search import add_line_numbers

    doc = _locked(lambda: store.find_document_by_identifier(path))
    if doc is None:
        raise ResourceNotFoundError(f"Document not found: {path}")
    return add_line_numbers(doc["doc"])


def build_server(store: Store) -> MCPServer:
    server = MCPServer("qmd", instructions=build_instructions(store), version=_package_version())
    start_time = time.monotonic()

    @server.tool()
    def status() -> CallToolResult:
        """Show the status of the qmd index: collections, document counts, and embedding coverage."""
        return _status_impl(store)

    @server.tool()
    def get(
        file: str,
        from_line: int | None = None,
        max_lines: int | None = None,
        line_numbers: bool = True,
    ) -> CallToolResult:
        """Retrieve the full content of a document by its file path or docid (#abc123).
        Supports a line-range suffix: 'notes/a.md:100' starts at line 100;
        'notes/a.md:100:40' reads 40 lines from line 100."""
        return _get_impl(store, file, from_line, max_lines, line_numbers)

    @server.tool()
    def multi_get(
        pattern: str,
        max_lines: int | None = None,
        max_bytes: int = 65536,
        line_numbers: bool = True,
    ) -> CallToolResult:
        """Retrieve multiple documents by a comma-separated list of docids/paths.
        Skips files larger than max_bytes."""
        return _multi_get_impl(store, pattern, max_lines, max_bytes, line_numbers)

    @server.tool()
    def query(
        query: str | None = None,
        limit: int = 10,
        min_score: float = 0.0,
        collections: list[str] | None = None,
        intent: str | None = None,
        rerank: bool = True,
        filter: dict | None = None,
        searches: list | None = None,
        # StrictInt, not int: pydantic's lax int coerces JSON true/false to 1/0
        # and "10" to 10 (CLAUDE.md bool-first rule; Node's zod z.number()
        # rejects all three). The schema still advertises a plain integer.
        candidateLimit: StrictInt | None = None,
    ) -> CallToolResult:
        """Search the knowledge base using hybrid keyword+semantic search with reranking.
        Always provide `intent` to sharpen ranking and snippets. Scope to specific
        collections with `collections`; omit to search all collections. Use `filter` to
        constrain results by document metadata (recursive JSON AST: and/or/not, eq/ne/gt/
        gte/lt/lte, in/nin/all, exists). Example:
        {"key": "status", "operator": "eq", "value": "published"}"""
        from pyqmd_mlx.mcp._errors import error_result

        if searches:
            if query:
                return error_result(
                    "Error: 'query' and 'searches' are mutually exclusive; provide only one"
                )
            return error_result(
                "Error: typed 'searches' are not supported; provide plain-text 'query' instead"
            )
        if not query:
            return error_result(
                "Error: provide either 'query' (plain text) or 'searches' (typed sub-queries)"
            )
        if candidateLimit is not None and candidateLimit < 1:
            # A negative LIMIT means "no limit" to SQLite, and 0 silently
            # returns nothing -- neither is a meaningful candidate pool.
            return error_result("Error: 'candidateLimit' must be a positive integer")
        return _query_impl(
            store,
            query,
            limit,
            min_score,
            collections,
            intent,
            rerank,
            filter,
            candidate_limit=candidateLimit,
        )

    @server.resource("qmd://{+path}")
    def read_document(path: str) -> str:
        """A markdown document from your qmd knowledge base. Use the query/get/multi_get
        tools to discover documents."""
        return _read_document_impl(store, path)

    @server.custom_route("/health", methods=["GET"])
    async def health(request):
        from starlette.responses import JSONResponse

        return JSONResponse({"status": "ok", "uptime": int(time.monotonic() - start_time)})

    return server


def run_stdio(db_path: str | None = None) -> None:
    from pyqmd_mlx.cli._db import get_store

    store = get_store(db_path)
    try:
        server = build_server(store)
        server.run(transport="stdio")
    finally:
        _locked(store.close)


def normalize_origin(origin: str) -> str | None:
    """Reduce an Origin header to its `scheme://host[:port]` form for
    allowlist comparison (Node origin-guard.ts normalizeOrigin: `new
    URL(origin).origin`). Returns None when the value is unusable --
    wrong scheme or unparseable -- which never matches."""
    try:
        parsed = urlparse(origin.strip())
        # `.port` validates lazily -- a non-numeric or out-of-range port
        # raises here, not in urlparse, so it belongs inside the try.
        port = parsed.port
    except ValueError:
        return None
    if parsed.scheme not in ("http", "https"):
        return None
    hostname = parsed.hostname
    if not hostname:
        return None
    host_part = f"[{hostname}]" if ":" in hostname else hostname
    default_port = 443 if parsed.scheme == "https" else 80
    if port and port != default_port:
        return f"{parsed.scheme}://{host_part}:{port}".lower()
    return f"{parsed.scheme}://{host_part}".lower()


def _header_hostname(value: str) -> str | None:
    """Parse a Host header (or bare hostname) to its hostname, or None when
    unparseable (Node origin-guard.ts hostHeaderHostname). A malformed value
    is untrusted, never a crash."""
    try:
        return urlparse(f"http://{value.strip()}").hostname
    except ValueError:
        return None


def is_loopback_hostname(hostname: str) -> bool:
    h = hostname.strip().lower().removeprefix("[").removesuffix("]")
    if h == "localhost" or h.endswith(".localhost"):
        return True
    if h in ("::1", "0:0:0:0:0:0:0:1"):
        return True
    v4 = h[7:] if h.startswith("::ffff:") else h
    parts = v4.split(".")
    return (
        len(parts) == 4
        # isascii+isdecimal, not isdigit: isdigit() accepts '²', which int() rejects.
        and all(p.isascii() and p.isdecimal() and 0 <= int(p) <= 255 for p in parts)
        and v4.startswith("127.")
    )


WILDCARD_BINDS = {"", "0.0.0.0", "::", "[::]", "*"}


def resolve_guard(
    host: str, allowed_origins: list[str] | None, allowed_hosts: list[str] | None
) -> dict:
    if allowed_origins is not None and "*" in [o.strip() for o in allowed_origins]:
        return {"disabled": True}
    hosts = [h.strip().lower() for h in (allowed_hosts or [])]
    bind = host.strip().lower()
    if bind not in WILDCARD_BINDS and not is_loopback_hostname(bind):
        # Explicit non-loopback interface (Node origin-guard.ts
        # resolveOriginGuard): its own address is a legitimate Host, no
        # allowlist entry needed.
        if bind not in hosts:
            hosts.append(bind)
    # Normalize allowlist origins the same way request Origins are
    # normalized (Node: `.map(normalizeOrigin).filter(Boolean)`); entries
    # that are not http(s) origins can never match and are dropped.
    origins = []
    for o in allowed_origins or []:
        normalized = normalize_origin(o)
        if normalized is not None and normalized not in origins:
            origins.append(normalized)
    return {
        "disabled": False,
        "allowed_origins": origins,
        "allowed_hosts": hosts,
        "enforce_host": bind not in WILDCARD_BINDS or bool(allowed_hosts),
    }


def check_origin(
    origin: str | None, host_header: str | None, guard: dict
) -> tuple[bool, str | None]:
    if guard.get("disabled"):
        return True, None
    if origin:
        # Only http(s) Origins participate (Node origin-guard.ts
        # originHostname); anything else is untrusted unless allowlisted,
        # and an unparseable value is untrusted, never a crash.
        normalized = normalize_origin(origin)
        # normalize_origin already validated this string, so the hostname
        # parse below cannot raise.
        hostname = urlparse(normalized).hostname if normalized is not None else None
        if not (
            (hostname is not None and is_loopback_hostname(hostname))
            or (normalized is not None and normalized in guard.get("allowed_origins", []))
        ):
            return False, f"Origin not allowed: {origin}"
    # Missing Origin is allowed (non-browser clients omit it).
    if guard.get("enforce_host") and host_header:
        hostname = _header_hostname(host_header)
        if hostname is None:
            return False, f"Host not allowed: {host_header}"
        if (
            not is_loopback_hostname(hostname)
            and host_header.strip().lower() not in guard.get("allowed_hosts", [])
            and hostname.lower() not in guard.get("allowed_hosts", [])
        ):
            return False, f"Host not allowed: {host_header}"
    return True, None


def check_request_origin(headers: dict, guard: dict) -> tuple[bool, str | None]:
    """Dict-signature adapter over check_origin for ASGI header maps."""
    lowered = {str(k).lower(): v for k, v in headers.items()}
    return check_origin(lowered.get("origin"), lowered.get("host"), guard)


def _origin_guard_wrapper(app, guard: dict):
    """Thin ASGI wrapper applying the Node-parity verdict ahead of routing.

    This wrapper is the ONLY guard: the SDK TransportSecuritySettings is
    built with DNS-rebinding protection disabled (see build_http_app),
    because a second exact-match check underneath can only refuse more --
    it can never allow what this wrapper allows (Node origin-guard.ts has
    no second layer). Rejects return Node's -32003 JSON-RPC failure body.
    """

    async def _guarded(scope, receive, send):
        scope_type = scope.get("type")
        if scope_type == "websocket":
            # Fail closed: pyqmd (like Node) serves no websocket routes, so a
            # handshake is refused here (1008 = policy violation; closing
            # before accept makes the server answer the upgrade with 403)
            # instead of trusting the router to have nothing to match.
            from starlette.websockets import WebSocketClose

            await WebSocketClose(code=1008)(scope, receive, send)
            return
        if scope_type not in ("http", "lifespan"):
            return
        if scope_type == "http":
            headers = {
                k.decode("latin-1").lower(): v.decode("latin-1")
                for k, v in scope.get("headers", [])
            }
            ok, reason = check_request_origin(headers, guard)
            if not ok:
                from starlette.responses import JSONResponse

                # Node parity (server.ts): a log breadcrumb on guard reject.
                print(
                    f"{scope.get('method', '')} {scope.get('path', '')} 403 — {reason}",
                    file=sys.stderr,
                )
                response = JSONResponse(
                    {
                        "jsonrpc": "2.0",
                        "error": {"code": -32003, "message": f"Forbidden: {reason}"},
                        "id": None,
                    },
                    status_code=403,
                )
                await response(scope, receive, send)
                return
        await app(scope, receive, send)

    return _guarded


def build_http_app(
    store: Store,
    host: str = "localhost",
    port: int = 8181,
    allowed_origins: list[str] | None = None,
    allowed_hosts: list[str] | None = None,
):
    """Build the Streamable HTTP ASGI app without binding a real network
    listener -- used by tests (via starlette.testclient.TestClient) and by
    run_http (which binds a real listener around the same app)."""
    from mcp.server.transport_security import TransportSecuritySettings

    server = build_server(store)
    # The _origin_guard_wrapper below is the only guard (Node has no second
    # layer): SDK DNS-rebinding protection stays off so its exact-match
    # middleware can never refuse what the wrapper allows. QMD_ALLOWED_*
    # allowlists feed resolve_guard, not the SDK.
    security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    app = server.streamable_http_app(host=host, transport_security=security, stateless_http=True)
    guard = resolve_guard(host, allowed_origins, allowed_hosts)
    return _origin_guard_wrapper(app, guard)


def _shutdown_breadcrumb(server) -> None:
    """Chain a Node-style `Shutting down (SIG...)...` stderr breadcrumb onto
    uvicorn's exit handler (Node server.ts SIGTERM/SIGINT handlers).

    uvicorn installs `server.handle_exit` for SIGINT/SIGTERM when `serve()`
    runs in the main thread (`capture_signals` binds whatever the attribute
    holds at that point), so patching the instance here adds the breadcrumb
    without touching signal machinery -- graceful shutdown, ordered close,
    and re-raise behavior all stay uvicorn's."""
    original = server.handle_exit

    def _handle_exit_with_breadcrumb(sig, frame):
        try:
            name = signal.Signals(sig).name
        except ValueError:
            name = str(sig)
        print(f"Shutting down ({name})...", file=sys.stderr)
        original(sig, frame)

    server.handle_exit = _handle_exit_with_breadcrumb


class PortInUseError(OSError):
    """The HTTP bind address is already taken. Carries Node's message
    (qmd.ts: `Port N already in use. Try a different port with --port.`);
    the CLI layer turns it into exit 1 -- server code never exits."""

    def __init__(self, host: str, port: int):
        super().__init__(f"Port {port} already in use. Try a different port with --port.")
        self.bind_host = host
        self.bind_port = port


def _log_startup_warnings(
    host: str,
    port: int,
    allowed_origins: list[str] | None,
    allowed_hosts: list[str] | None,
) -> None:
    """Mirror Node's startup log lines (server.ts): the listening address
    plus DNS-rebinding warnings when protection is weakened. All stderr."""
    origins = allowed_origins or []
    if "*" in [o.strip() for o in origins]:
        print(
            "Warning: QMD_ALLOWED_ORIGINS=* — DNS-rebinding protection is off. "
            "Only do this behind your own authenticating proxy.",
            file=sys.stderr,
        )
    elif host.strip().lower() in WILDCARD_BINDS and not allowed_hosts:
        print(
            f"Warning: bound to {host} with no QMD_ALLOWED_HOSTS — Host validation "
            "is off and the index is readable by anyone who can reach this port.",
            file=sys.stderr,
        )
    print(f"QMD MCP server listening on http://{host}:{port}/mcp", file=sys.stderr)


def _bind_sockets(host: str, port: int) -> list:
    """Bind the listen socket(s) ourselves (SO_REUSEADDR, like uvicorn) and
    hand them to uvicorn via Server.serve(sockets=...) instead of probing.

    A probe without SO_REUSEADDR disagrees with the server about a recently
    used port (TIME_WAIT after a disconnect reads as "in use" for ~30s);
    binding the real sockets up front removes the race and the disagreement.

    A hostname binds every address it resolves to, so `localhost` listens
    on both 127.0.0.1 and ::1 -- a superset of Node, whose listen() takes
    only the first resolved address (::1 on macOS). An address literal
    (`127.0.0.1`, `::1`, `0.0.0.0`, `::`) binds exactly that address.
    Brackets are stripped so '[::1]' binds instead of crashing getaddrinfo.

    Raises PortInUseError if any address is taken (errno 48/98, darwin/linux
    EADDRINUSE) -- never a half-bound server."""
    import socket

    clean = host.strip().removeprefix("[").removesuffix("]")
    addresses = []
    for family, _, _, _, sockaddr in socket.getaddrinfo(
        clean or None, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE
    ):
        if (family, sockaddr[0]) not in addresses:
            addresses.append((family, sockaddr[0]))
    dual_stack = {family for family, _ in addresses} >= {socket.AF_INET, socket.AF_INET6}

    socks: list[socket.socket] = []
    try:
        for family, address in addresses:
            sock = socket.socket(family, socket.SOCK_STREAM)
            socks.append(sock)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if family == socket.AF_INET6 and dual_stack:
                # The IPv4 address gets its own socket; without V6ONLY a
                # v4-mapped wildcard v6 socket would collide with it.
                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            sock.bind((address, port))
            # Port 0 means "pick one": every family must share that pick.
            port = sock.getsockname()[1]
    except OSError as exc:
        for sock in socks:
            sock.close()
        if exc.errno in (48, 98):
            raise PortInUseError(host, port) from exc
        raise
    return socks


def run_http(
    host: str = "localhost",
    port: int = 8181,
    db_path: str | None = None,
    allowed_origins: list[str] | None = None,
    allowed_hosts: list[str] | None = None,
) -> None:
    import anyio
    import uvicorn

    from pyqmd_mlx.cli._db import get_store

    # Bind before opening the store: a conflicting bind is a startup usage
    # error, not a serving failure. The bound socket goes straight into
    # uvicorn (Server.serve(sockets=...), the gunicorn-worker path), so the
    # server listens on exactly what was bound -- no probe-then-bind race.
    socks = _bind_sockets(host, port)
    try:
        store = get_store(db_path)
        app = build_http_app(
            store,
            host=host,
            port=port,
            allowed_origins=allowed_origins,
            allowed_hosts=allowed_hosts,
        )
    except BaseException:
        # The sockets are still ours: uvicorn only takes ownership inside
        # serve(). Close them here so a mid-startup failure cannot leak fds.
        for sock in socks:
            sock.close()
        raise
    try:
        # Serve the same guarded app the tests exercise: previously run_http
        # went through SDK `server.run(transport="streamable-http")`, which
        # bypassed the Task 1 origin-guard pre-check on the real listener.
        # The uvicorn driving below mirrors what the SDK does internally
        # (MCPServer.run_streamable_http_async): same ASGI app shape,
        # stateless_http=True (inside build_http_app), same log level.
        # uvicorn closes the handed-in sockets on shutdown.
        actual_port = socks[0].getsockname()[1]
        _log_startup_warnings(host, actual_port, allowed_origins, allowed_hosts)
        config = uvicorn.Config(app, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)
        _shutdown_breadcrumb(server)
        anyio.run(server.serve, socks)
    finally:
        _locked(store.close)
