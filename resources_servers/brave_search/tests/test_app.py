from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from nemo_gym.server_utils import ServerClient
from resources_servers.brave_search.app import (
    BaseSearchQueryRequest,
    BraveSearchResourcesServer,
    BraveSearchResourcesServerConfig,
    box_parser,
)


class TestApp:
    def _config(self) -> BraveSearchResourcesServerConfig:
        return BraveSearchResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            brave_api_key="dummy_key",  # pragma: allowlist secret
        )

    def test_sanity(self) -> None:
        BraveSearchResourcesServer(config=self._config(), server_client=MagicMock(spec=ServerClient))

    def test_box_parser_valid_content(self) -> None:
        result = box_parser("The answer is \\boxed{42}")
        assert result == "42"

        result = box_parser("After calculation: \\boxed{x + y = 10}")
        assert result == "x + y = 10"

        result = box_parser("No boxed content here")
        assert result is None

        result = box_parser("")
        assert result is None

    async def test_search_dispatches_to_mcp_tool(self) -> None:
        server = BraveSearchResourcesServer(config=self._config(), server_client=MagicMock(spec=ServerClient))
        server._mcp_session = MagicMock()
        server._mcp_session.call_tool = AsyncMock(
            return_value=SimpleNamespace(
                isError=False,
                content=[SimpleNamespace(text="[1] Example (example.com)\n    URL: https://example.com\n")],
            )
        )

        response = await server.search(BaseSearchQueryRequest(query="test query"))

        server._mcp_session.call_tool.assert_called_once_with(
            "brave_web_search", {"query": "test query", "count": server.config.max_results}
        )
        assert "example.com" in response.search_results

    async def test_search_mcp_error(self) -> None:
        server = BraveSearchResourcesServer(config=self._config(), server_client=MagicMock(spec=ServerClient))
        server._mcp_session = MagicMock()
        server._mcp_session.call_tool = AsyncMock(
            return_value=SimpleNamespace(isError=True, content=[SimpleNamespace(text="rate limited")])
        )

        response = await server.search(BaseSearchQueryRequest(query="test query"))

        assert response.search_results.startswith("Error:")
        assert "rate limited" in response.search_results

    async def test_search_unexpected_exception(self) -> None:
        server = BraveSearchResourcesServer(config=self._config(), server_client=MagicMock(spec=ServerClient))
        server._mcp_session = MagicMock()
        server._mcp_session.call_tool = AsyncMock(side_effect=RuntimeError("boom"))

        response = await server.search(BaseSearchQueryRequest(query="test query"))

        assert response.search_results.startswith("Error:")
        assert "boom" in response.search_results
