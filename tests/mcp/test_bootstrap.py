def test_qmd_mcp_package_is_importable():
    import pyqmd_mlx.mcp  # noqa: F401


def test_official_mcp_sdk_mcpserver_is_importable():
    from mcp.server.mcpserver import MCPServer  # noqa: F401


def test_official_mcp_sdk_result_types_are_importable():
    from mcp.types import (  # noqa: F401
        CallToolResult,
        EmbeddedResource,
        TextContent,
        TextResourceContents,
    )
