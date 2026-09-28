from typer.testing import CliRunner

from pyqmd_mlx.cli.commands.mcp import _parse_csv_env, app

runner = CliRunner()


def test_mcp_command_invokes_run_stdio(monkeypatch):
    called = {}
    monkeypatch.setattr(
        "pyqmd_mlx.cli.commands.mcp.run_stdio", lambda: called.setdefault("ran", True)
    )
    result = runner.invoke(app, ["mcp"])
    assert result.exit_code == 0
    assert called.get("ran") is True


def test_mcp_command_http_flag_invokes_run_http(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "pyqmd_mlx.cli.commands.mcp.run_http", lambda **kwargs: captured.update(kwargs)
    )
    monkeypatch.delenv("QMD_HOST", raising=False)
    result = runner.invoke(app, ["mcp", "--http", "--port", "9001"])
    assert result.exit_code == 0
    assert captured["port"] == 9001
    assert captured["host"] == "localhost"


def test_mcp_command_http_bare_flag_uses_node_defaults(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "pyqmd_mlx.cli.commands.mcp.run_http", lambda **kwargs: captured.update(kwargs)
    )
    monkeypatch.delenv("QMD_HOST", raising=False)
    result = runner.invoke(app, ["mcp", "--http"])
    assert result.exit_code == 0
    assert captured["port"] == 8181
    assert captured["host"] == "localhost"


def test_mcp_command_reads_allowed_origins_and_hosts_env_vars(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "pyqmd_mlx.cli.commands.mcp.run_http", lambda **kwargs: captured.update(kwargs)
    )
    monkeypatch.setenv("QMD_ALLOWED_ORIGINS", "https://a.com, https://b.com")
    monkeypatch.setenv("QMD_ALLOWED_HOSTS", "a.com,b.com")
    result = runner.invoke(app, ["mcp", "--http"])
    assert result.exit_code == 0
    assert captured["allowed_origins"] == ["https://a.com", "https://b.com"]
    assert captured["allowed_hosts"] == ["a.com", "b.com"]


def test_mcp_command_without_env_vars_passes_none_for_allowlists(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "pyqmd_mlx.cli.commands.mcp.run_http", lambda **kwargs: captured.update(kwargs)
    )
    monkeypatch.delenv("QMD_ALLOWED_ORIGINS", raising=False)
    monkeypatch.delenv("QMD_ALLOWED_HOSTS", raising=False)
    result = runner.invoke(app, ["mcp", "--http"])
    assert result.exit_code == 0
    assert captured["allowed_origins"] is None
    assert captured["allowed_hosts"] is None


def test_parse_csv_env_returns_none_for_unset(monkeypatch):
    monkeypatch.delenv("QMD_ALLOWED_ORIGINS", raising=False)
    assert _parse_csv_env("QMD_ALLOWED_ORIGINS") is None


def test_parse_csv_env_returns_none_for_blank(monkeypatch):
    monkeypatch.setenv("QMD_ALLOWED_ORIGINS", "   ")
    assert _parse_csv_env("QMD_ALLOWED_ORIGINS") is None


def test_parse_csv_env_returns_none_for_comma_only_malformed_value(monkeypatch):
    monkeypatch.setenv("QMD_ALLOWED_ORIGINS", ",")
    assert _parse_csv_env("QMD_ALLOWED_ORIGINS") is None


def test_parse_csv_env_returns_none_for_whitespace_and_commas_only(monkeypatch):
    monkeypatch.setenv("QMD_ALLOWED_ORIGINS", " , , ")
    assert _parse_csv_env("QMD_ALLOWED_ORIGINS") is None


def test_parse_csv_env_returns_parsed_list_for_populated_value(monkeypatch):
    monkeypatch.setenv("QMD_ALLOWED_ORIGINS", "https://a.com, https://b.com")
    assert _parse_csv_env("QMD_ALLOWED_ORIGINS") == ["https://a.com", "https://b.com"]


def test_mcp_http_resolves_host_from_qmd_host_env(monkeypatch):
    from typer.testing import CliRunner

    from pyqmd_mlx.cli.commands.mcp import app

    captured = {}
    monkeypatch.setattr(
        "pyqmd_mlx.cli.commands.mcp.run_http", lambda **kwargs: captured.update(kwargs)
    )
    monkeypatch.setenv("QMD_HOST", "0.0.0.0")
    result = CliRunner().invoke(app, ["mcp", "--http"])
    assert result.exit_code == 0
    assert captured["host"] == "0.0.0.0"


def test_mcp_http_flag_host_overrides_qmd_host_env(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "pyqmd_mlx.cli.commands.mcp.run_http", lambda **kwargs: captured.update(kwargs)
    )
    monkeypatch.setenv("QMD_HOST", "localhost")
    result = runner.invoke(app, ["mcp", "--http", "--host", "0.0.0.0"])
    assert result.exit_code == 0
    assert captured["host"] == "0.0.0.0"


def test_mcp_http_blank_qmd_host_env_falls_back_to_default(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "pyqmd_mlx.cli.commands.mcp.run_http", lambda **kwargs: captured.update(kwargs)
    )
    monkeypatch.setenv("QMD_HOST", "   ")
    result = runner.invoke(app, ["mcp", "--http"])
    assert result.exit_code == 0
    assert captured["host"] == "localhost"


def test_mcp_http_occupied_port_exits_1_with_clean_message():
    import socket

    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", 0))
    holder.listen(1)
    try:
        port = holder.getsockname()[1]
        result = runner.invoke(app, ["mcp", "--http", "--port", str(port)])
    finally:
        holder.close()
    assert result.exit_code == 1
    combined = result.output + (result.stderr or "")
    assert f"Port {port} already in use. Try a different port with --port." in combined


def test_mcp_http_port_in_use_error_from_server_exits_1(monkeypatch):
    """The CLI translates run_http's PortInUseError (raised from the real
    bind, not a probe) into Node's one-liner + exit 1."""
    from pyqmd_mlx.mcp.server import PortInUseError

    def _raise(**kwargs):
        raise PortInUseError("127.0.0.1", 9999)

    monkeypatch.setattr("pyqmd_mlx.cli.commands.mcp.run_http", _raise)
    result = runner.invoke(app, ["mcp", "--http"])
    assert result.exit_code == 1
    combined = result.output + (result.stderr or "")
    assert "Port 9999 already in use. Try a different port with --port." in combined
