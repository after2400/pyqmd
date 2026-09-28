"""Proves the HTTP transport's origin-guard genuinely rejects a disallowed
request, not just that it accepts a valid one -- required by the design
spec's testing strategy. Targets POST /mcp directly, since the Node-parity
pre-check in build_http_app gates every request ahead of routing (mirroring
origin-guard.ts): a disallowed Origin or Host returns 403 with a JSON-RPC
-32003 body. It is the only guard -- SDK DNS-rebinding protection is off.
"""

import pytest
from starlette.testclient import TestClient

from pyqmd_mlx.mcp.server import build_http_app
from pyqmd_mlx.store import Store

_PING = {"jsonrpc": "2.0", "method": "ping", "id": 1}
_HEADERS = {"Accept": "application/json, text/event-stream"}


def _post_ping(client, origin="http://127.0.0.1:8181"):
    headers = dict(_HEADERS)
    if origin is not None:
        headers["Origin"] = origin
    return client.post("/mcp", json=_PING, headers=headers)


def test_wrapper_allows_bracketed_ipv6_host_through_app():
    """Node allows any loopback Host (origin-guard.ts isLoopbackHostname).
    Exercised at ASGI-scope level rather than via TestClient, whose own
    host:port splitter cannot parse bracketed IPv6 base URLs (starlette
    testclient ValueError) -- the header bytes below are exactly what a
    real listener delivers for Host: [::1]:8181."""
    import asyncio

    from starlette.responses import JSONResponse

    from pyqmd_mlx.mcp.server import _origin_guard_wrapper, resolve_guard

    async def inner(scope, receive, send):
        await JSONResponse({"ok": True})(scope, receive, send)

    async def go():
        app = _origin_guard_wrapper(inner, resolve_guard("localhost", None, None))
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp",
            "headers": [(b"host", b"[::1]:8181")],
        }
        messages = []

        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            messages.append(message)

        await app(scope, receive, send)
        return messages

    messages = asyncio.run(go())
    assert messages[0]["status"] not in (403, 421)


def test_wrapper_allows_unlisted_host_on_wildcard_bind():
    """Wildcard bind (0.0.0.0) enforces Host only with an explicit allowlist;
    a container hostname must pass the app, not 421 at the SDK layer."""
    store = Store(":memory:")
    app = build_http_app(store, host="0.0.0.0", port=8181)

    with TestClient(app, base_url="http://mycontainer:8181") as client:
        response = _post_ping(client)

    assert response.status_code not in (403, 421)


def test_wrapper_allows_127_range_host_through_app():
    """127.0.0.0/8 (not just 127.0.0.1) is loopback per Node; must not 421."""
    store = Store(":memory:")
    app = build_http_app(store, host="localhost", port=8181)

    with TestClient(app, base_url="http://127.0.0.5:8181") as client:
        response = _post_ping(client)

    assert response.status_code not in (403, 421)


def test_loopback_origins_allowed_through_app():
    """*.localhost and 127/8 Origins are loopback; the SDK layer answers
    them with a plain 403, so asserting the allow proves wrapper-only."""
    store = Store(":memory:")
    app = build_http_app(store, host="localhost", port=8181)

    with TestClient(app, base_url="http://localhost:8181") as client:
        for origin in ("http://foo.localhost:8181", "http://127.0.0.5:8181"):
            response = _post_ping(client, origin=origin)
            assert response.status_code not in (403, 421)


def test_explicit_hosts_allowlist_does_not_lock_out_localhost():
    """Setting QMD_ALLOWED_HOSTS-equivalent must extend, not replace, the
    loopback trust: localhost still passes through the app."""
    store = Store(":memory:")
    app = build_http_app(store, host="localhost", port=8181, allowed_hosts=["mycontainer"])

    with TestClient(app, base_url="http://localhost:8181") as client:
        response = _post_ping(client)

    assert response.status_code not in (403, 421)


def test_malformed_origin_returns_403_not_500():
    """An unparseable Origin (bad IPv6 bracket) is untrusted, not a crash:
    Node's originHostname returns undefined and the verdict is Forbidden."""
    store = Store(":memory:")
    app = build_http_app(store, host="localhost", port=8181)

    with TestClient(app, base_url="http://localhost:8181") as client:
        response = _post_ping(client, origin="http://[::1")

    assert response.status_code == 403
    assert response.json()["error"]["code"] == -32003


@pytest.mark.parametrize("origin", ["http://localhost:abc", "http://localhost:99999"])
def test_malformed_origin_port_returns_403_not_500(origin):
    """urlparse defers port validation to `.port`, which raises -- that read
    must stay inside the untrusted-value path, not escape as a 500."""
    store = Store(":memory:")
    app = build_http_app(store, host="localhost", port=8181)

    with TestClient(app, base_url="http://localhost:8181", raise_server_exceptions=False) as c:
        response = _post_ping(c, origin=origin)

    assert response.status_code == 403
    assert response.json()["error"]["code"] == -32003


def test_non_ascii_digit_octets_are_untrusted_not_a_crash():
    """str.isdigit() accepts '²', which int() then rejects: a Host or Origin
    like 127.0.0.² must be refused, never raise."""
    from pyqmd_mlx.mcp.server import check_origin, is_loopback_hostname, resolve_guard

    assert is_loopback_hostname("127.0.0.²") is False
    guard = resolve_guard("localhost", None, None)
    assert check_origin(None, "127.0.0.²:8181", guard)[0] is False
    assert check_origin("http://127.0.0.²:8181", None, guard)[0] is False


def test_malformed_allowlist_origin_is_dropped_not_a_startup_crash():
    from pyqmd_mlx.mcp.server import resolve_guard

    guard = resolve_guard("localhost", ["https://ok.example", "https://bad.example:abc"], None)
    assert guard["allowed_origins"] == ["https://ok.example"]


def test_allowlisted_origin_with_trailing_slash_matches():
    """Allowlist entries are origins; a trailing slash must not break the
    match (Node normalizes via URL.origin)."""
    store = Store(":memory:")
    app = build_http_app(
        store,
        host="localhost",
        port=8181,
        allowed_origins=["https://qmd.internal.example.com/"],
    )

    with TestClient(app, base_url="http://localhost:8181") as client:
        response = _post_ping(client, origin="https://qmd.internal.example.com/some/page")

    assert response.status_code not in (403, 421)


def test_non_http_loopback_origin_is_rejected():
    """Node only trusts http(s) Origins: ftp://localhost is refused even
    though the hostname is loopback (origin-guard.ts originHostname)."""
    store = Store(":memory:")
    app = build_http_app(store, host="localhost", port=8181)

    with TestClient(app, base_url="http://localhost:8181") as client:
        response = _post_ping(client, origin="ftp://localhost/file")

    assert response.status_code == 403
    assert response.json()["error"]["code"] == -32003


def test_valid_host_and_origin_are_not_rejected_by_origin_guard():
    store = Store(":memory:")
    app = build_http_app(store, host="127.0.0.1", port=8123)

    with TestClient(app, base_url="http://127.0.0.1:8123") as client:
        response = client.post(
            "/mcp", json=_PING, headers={**_HEADERS, "Origin": "http://127.0.0.1:8123"}
        )

    assert response.status_code not in (403, 421)


def test_disallowed_host_is_rejected_with_32003():
    store = Store(":memory:")
    app = build_http_app(store, host="127.0.0.1", port=8123)

    with TestClient(app, base_url="http://evil.example.com") as client:
        response = client.post(
            "/mcp", json=_PING, headers={**_HEADERS, "Origin": "http://127.0.0.1:8123"}
        )

    assert response.status_code == 403
    assert response.json()["error"]["code"] == -32003


def test_disallowed_origin_is_rejected_with_403():
    store = Store(":memory:")
    app = build_http_app(store, host="127.0.0.1", port=8123)

    with TestClient(app, base_url="http://127.0.0.1:8123") as client:
        response = client.post(
            "/mcp", json=_PING, headers={**_HEADERS, "Origin": "http://evil.example.com"}
        )

    assert response.status_code == 403


def test_explicit_allowlist_permits_a_non_default_host():
    store = Store(":memory:")
    app = build_http_app(
        store,
        host="0.0.0.0",
        port=8123,
        allowed_hosts=["qmd.internal.example.com"],
        allowed_origins=["https://qmd.internal.example.com"],
    )

    with TestClient(app, base_url="http://qmd.internal.example.com") as client:
        response = client.post(
            "/mcp",
            json=_PING,
            headers={**_HEADERS, "Origin": "https://qmd.internal.example.com"},
        )

    assert response.status_code not in (403, 421)


def test_loopback_origin_variants_are_allowed():
    from pyqmd_mlx.mcp.server import is_loopback_hostname

    assert is_loopback_hostname("localhost") is True
    assert is_loopback_hostname("foo.localhost") is True
    assert is_loopback_hostname("127.0.0.5") is True
    assert is_loopback_hostname("::1") is True
    assert is_loopback_hostname("::ffff:127.0.0.1") is True
    assert is_loopback_hostname("0.0.0.0") is False
    assert is_loopback_hostname("evil.example.com") is False


def test_disallowed_origin_returns_jsonrpc_32003_body():
    from starlette.testclient import TestClient

    from pyqmd_mlx.mcp.server import build_http_app
    from pyqmd_mlx.store import Store

    store = Store(":memory:")
    app = build_http_app(store, host="127.0.0.1", port=8123)
    with TestClient(app, base_url="http://127.0.0.1:8123") as client:
        response = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "ping", "id": 1},
            headers={
                "Accept": "application/json, text/event-stream",
                "Origin": "http://evil.example.com",
            },
        )
    assert response.status_code == 403
    assert response.json()["error"]["code"] == -32003


def test_star_origin_bypass_disables_all_checks():
    from pyqmd_mlx.mcp.server import resolve_guard

    assert resolve_guard("127.0.0.1", ["*"], None) == {"disabled": True}

    store = Store(":memory:")
    app = build_http_app(store, host="127.0.0.1", port=8123, allowed_origins=["*"])

    with TestClient(app, base_url="http://127.0.0.1:8123") as client:
        response = client.post(
            "/mcp", json=_PING, headers={**_HEADERS, "Origin": "http://evil.example.com"}
        )

    assert response.status_code not in (403, 421)


def test_missing_origin_is_allowed():
    store = Store(":memory:")
    app = build_http_app(store, host="127.0.0.1", port=8123)

    with TestClient(app, base_url="http://127.0.0.1:8123") as client:
        response = client.post("/mcp", json=_PING, headers=dict(_HEADERS))

    assert response.status_code not in (403, 421)


def test_wildcard_bind_skips_host_check_but_keeps_origin_check():
    store = Store(":memory:")
    app = build_http_app(store, host="0.0.0.0", port=8123)

    # "0.0.0.0" is a non-loopback hostname per is_loopback_hostname, so a
    # concrete bind would reject it as Host -- here it must pass, proving Host
    # enforcement is off for the wildcard bind.
    with TestClient(app, base_url="http://0.0.0.0:8123") as client:
        allowed = client.post("/mcp", json=_PING, headers=dict(_HEADERS))
        rejected = client.post(
            "/mcp", json=_PING, headers={**_HEADERS, "Origin": "http://evil.example.com"}
        )

    assert allowed.status_code not in (403, 421)
    assert rejected.status_code == 403
    assert rejected.json()["error"]["code"] == -32003


def test_non_loopback_bind_self_allows_own_host():
    # Node origin-guard.ts resolveOriginGuard: an explicit non-loopback bind
    # address is itself a legitimate Host, no allowlist entry needed.
    from pyqmd_mlx.mcp.server import check_origin, resolve_guard

    guard = resolve_guard("192.168.1.5", None, None)
    assert guard["allowed_hosts"] == ["192.168.1.5"]
    ok, reason = check_origin(None, "192.168.1.5:8123", guard)
    assert ok, reason


def test_guard_reject_emits_stderr_breadcrumb(capsys):
    # Spec section 3.1: Node logs a line on guard reject; the pre-check must
    # leave the same kind of breadcrumb on stderr.
    store = Store(":memory:")
    app = build_http_app(store, host="127.0.0.1", port=8123)

    with TestClient(app, base_url="http://127.0.0.1:8123") as client:
        response = client.post(
            "/mcp", json=_PING, headers={**_HEADERS, "Origin": "http://evil.example.com"}
        )

    assert response.status_code == 403
    assert "403 — Origin not allowed" in capsys.readouterr().err


@pytest.mark.parametrize("scope_type", ["websocket", "unknown"])
def test_wrapper_fails_closed_on_non_http_scopes(scope_type):
    """Only http and lifespan scopes reach the app. pyqmd (like Node) serves
    no websocket routes, so a websocket handshake is closed at the guard
    rather than relying on the router having nothing to match."""
    import asyncio

    from pyqmd_mlx.mcp.server import _origin_guard_wrapper, resolve_guard

    reached = []

    async def inner(scope, receive, send):
        reached.append(scope["type"])

    async def go():
        app = _origin_guard_wrapper(inner, resolve_guard("localhost", None, None))
        scope = {
            "type": scope_type,
            "path": "/mcp",
            "headers": [(b"host", b"evil.example.com"), (b"origin", b"http://evil.example.com")],
        }
        messages = []

        async def receive():
            return {"type": "websocket.connect"}

        async def send(message):
            messages.append(message)

        await app(scope, receive, send)
        return messages

    messages = asyncio.run(go())
    assert reached == []
    if scope_type == "websocket":
        assert messages == [{"type": "websocket.close", "code": 1008, "reason": ""}]


def test_wrapper_passes_lifespan_scope_through():
    import asyncio

    from pyqmd_mlx.mcp.server import _origin_guard_wrapper, resolve_guard

    reached = []

    async def inner(scope, receive, send):
        reached.append(scope["type"])

    app = _origin_guard_wrapper(inner, resolve_guard("localhost", None, None))
    asyncio.run(app({"type": "lifespan"}, None, None))
    assert reached == ["lifespan"]
