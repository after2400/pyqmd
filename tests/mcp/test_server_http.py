import pytest
from starlette.testclient import TestClient

from pyqmd_mlx.mcp.server import build_http_app
from pyqmd_mlx.store import Store


def test_build_http_app_serves_health_endpoint():
    store = Store(":memory:")
    app = build_http_app(store, host="127.0.0.1", port=8123)

    with TestClient(app) as client:
        response = client.get(
            "/health", headers={"Host": "127.0.0.1:8123", "Origin": "http://127.0.0.1:8123"}
        )

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_includes_uptime_seconds():
    from starlette.testclient import TestClient

    from pyqmd_mlx.mcp.server import build_http_app
    from pyqmd_mlx.store import Store

    store = Store(":memory:")
    app = build_http_app(store, host="127.0.0.1", port=8124)
    with TestClient(app) as client:
        response = client.get(
            "/health", headers={"Host": "127.0.0.1:8124", "Origin": "http://127.0.0.1:8124"}
        )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert isinstance(body["uptime"], int) and body["uptime"] >= 0


def test_shutdown_breadcrumb_logs_signal_name_and_delegates(capsys):
    import signal
    from types import SimpleNamespace

    from pyqmd_mlx.mcp import server as server_mod

    calls = []
    fake = SimpleNamespace(handle_exit=lambda sig, frame: calls.append(sig))
    server_mod._shutdown_breadcrumb(fake)
    fake.handle_exit(signal.SIGTERM, None)

    assert calls == [signal.SIGTERM]
    assert "Shutting down (SIGTERM)..." in capsys.readouterr().err


def _has_ipv6_localhost():
    import socket

    try:
        infos = socket.getaddrinfo("localhost", 0, type=socket.SOCK_STREAM)
    except OSError:
        return False
    return {i[0] for i in infos} >= {socket.AF_INET, socket.AF_INET6}


def _close_all(socks):
    for s in socks:
        s.close()


def test_bind_sockets_conflict_raises_port_in_use_with_node_message():
    import socket

    from pyqmd_mlx.mcp.server import PortInUseError, _bind_sockets

    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", 0))
    holder.listen(1)
    try:
        port = holder.getsockname()[1]
        try:
            _close_all(_bind_sockets("127.0.0.1", port))
        except PortInUseError as exc:
            assert str(exc) == (f"Port {port} already in use. Try a different port with --port.")
            assert exc.bind_port == port
        else:
            raise AssertionError("expected PortInUseError")
    finally:
        holder.close()


@pytest.mark.skipif(not _has_ipv6_localhost(), reason="localhost has no IPv6 address here")
def test_bind_sockets_localhost_conflict_on_one_family_is_port_in_use():
    """A hostname binds every family; if any one is taken, the whole bind is
    a clean PortInUseError, not a half-bound server."""
    import socket

    from pyqmd_mlx.mcp.server import PortInUseError, _bind_sockets

    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", 0))
    holder.listen(1)
    try:
        with pytest.raises(PortInUseError):
            _close_all(_bind_sockets("localhost", holder.getsockname()[1]))
    finally:
        holder.close()


def test_bind_sockets_rebind_while_port_is_in_time_wait():
    """Restart right after a client disconnects: the server-side close leaves
    the port in TIME_WAIT, which only an SO_REUSEADDR bind can reuse."""
    import errno
    import socket

    from pyqmd_mlx.mcp.server import _bind_sockets

    (first,) = _bind_sockets("127.0.0.1", 0)
    port = first.getsockname()[1]
    first.listen(1)
    client = socket.create_connection(("127.0.0.1", port))
    conn, _ = first.accept()
    # Server side closes first -> the server end of the connection (bound to
    # `port`) enters TIME_WAIT, exactly as after uvicorn shuts down.
    conn.close()
    first.close()
    client.close()

    # Control: prove TIME_WAIT is really there -- a bind without
    # SO_REUSEADDR (the old probe) is refused.
    plain = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(OSError) as excinfo:
            plain.bind(("127.0.0.1", port))
        assert excinfo.value.errno == errno.EADDRINUSE
    finally:
        plain.close()

    (second,) = _bind_sockets("127.0.0.1", port)
    try:
        assert second.getsockname()[1] == port
    finally:
        second.close()


@pytest.mark.skipif(not _has_ipv6_localhost(), reason="localhost has no IPv6 address here")
def test_bind_sockets_localhost_binds_ipv4_and_ipv6_on_one_port():
    """`localhost` listens on 127.0.0.1 and ::1: a superset of Node (which
    binds only the first resolved address), so both http://127.0.0.1 and
    http://[::1] clients connect."""
    import socket

    from pyqmd_mlx.mcp.server import _bind_sockets

    socks = _bind_sockets("localhost", 0)
    try:
        addrs = sorted((s.family, s.getsockname()[0]) for s in socks)
        assert addrs == [(socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")]
        # Port 0 resolves once; every family shares it.
        assert len({s.getsockname()[1] for s in socks}) == 1
    finally:
        _close_all(socks)


def test_bind_sockets_address_literal_binds_exactly_that_address():
    from pyqmd_mlx.mcp.server import _bind_sockets

    socks = _bind_sockets("127.0.0.1", 0)
    try:
        assert [s.getsockname()[0] for s in socks] == ["127.0.0.1"]
    finally:
        _close_all(socks)


def test_bind_sockets_strips_bracketed_ipv6_host():
    from pyqmd_mlx.mcp.server import _bind_sockets

    socks = _bind_sockets("[::1]", 0)
    try:
        assert [s.getsockname()[0] for s in socks] == ["::1"]
    finally:
        _close_all(socks)


def test_startup_warnings_cover_disabled_guard_and_wildcard_bind(capsys):
    from pyqmd_mlx.mcp.server import _log_startup_warnings

    _log_startup_warnings("localhost", 8181, ["*"], None)
    err = capsys.readouterr().err
    assert "QMD_ALLOWED_ORIGINS=*" in err
    assert "listening on http://localhost:8181/mcp" in err

    _log_startup_warnings("0.0.0.0", 8181, None, None)
    err = capsys.readouterr().err
    assert "no QMD_ALLOWED_HOSTS" in err

    _log_startup_warnings("localhost", 8181, None, None)
    err = capsys.readouterr().err
    assert "Warning" not in err
    assert "listening on http://localhost:8181/mcp" in err
