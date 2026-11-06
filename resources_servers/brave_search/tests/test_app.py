from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.config_types import ResourcesServerRef
from nemo_gym.server_utils import ServerClient
from resources_servers.brave_search.app import (
    BaseSearchQueryRequest,
    BraveSearchResourcesServer,
    BraveSearchResourcesServerConfig,
    BraveSearchVerifyRequest,
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
        # search() now attempts one reconnect+retry before giving up (see
        # [CUSTOM-SEARCH-MCP-RECONNECT] in app.py) -- mock _connect_mcp_session/_close_mcp_session
        # so this test exercises "both the original call and the retry fail" without spawning a
        # real MCP session.
        server = BraveSearchResourcesServer(config=self._config(), server_client=MagicMock(spec=ServerClient))
        server._mcp_session = MagicMock()
        server._mcp_session.call_tool = AsyncMock(side_effect=RuntimeError("boom"))
        server._close_mcp_session = AsyncMock()
        server._connect_mcp_session = AsyncMock()

        response = await server.search(BaseSearchQueryRequest(query="test query"))

        assert response.search_results.startswith("Error:")
        assert "boom" in response.search_results
        server._close_mcp_session.assert_awaited_once()
        server._connect_mcp_session.assert_awaited_once()

    def _verify_request(self, *, expected_answer: str, assistant_text: str, verifier_type=None):
        from nemo_gym.openai_utils import NeMoGymResponse

        response = NeMoGymResponse(
            id="resp_test",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "id": "msg_test",
                    "content": [{"annotations": [], "text": assistant_text, "type": "output_text"}],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )
        return BraveSearchVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "test question"}]},
            response=response,
            expected_answer=expected_answer,
            verifier_type=verifier_type,
        )

    # [CUSTOM-VERIFIER-DISPATCH 2026-11-06] Dispatch tests, mirroring ns_tools's own
    # test_verify_delegates_to_math_with_judge / test_verify_uses_default_verifier.
    async def test_verify_uses_default_box_parser_when_verifier_type_unset(self) -> None:
        server = BraveSearchResourcesServer(config=self._config(), server_client=MagicMock(spec=ServerClient))
        server.server_client.post = AsyncMock()  # must NOT be called on the default path

        request = self._verify_request(expected_answer="42", assistant_text="The answer is \\boxed{42}.")
        result = await server.verify(request)

        assert result.reward == 1.0
        assert result.parsed_option == "42"
        assert result.delegated_response is None
        server.server_client.post.assert_not_called()

    async def test_verify_delegates_to_configured_verifier(self) -> None:
        config = self._config()
        config.default_verifier = "box_parser"
        config.verifiers = {
            "equivalence_llm_judge": ResourcesServerRef(type="resources_servers", name="equivalence_llm_judge"),
        }
        server = BraveSearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={"reward": 1.0, "verdict_label": "[[A=B]]"})
        server.server_client.post = AsyncMock(return_value=mock_response)

        request = self._verify_request(
            expected_answer="Saarbrücken",
            assistant_text="The city is Saarbrücken.",
            verifier_type="equivalence_llm_judge",
        )
        result = await server.verify(request)

        assert result.reward == 1.0
        assert result.delegated_response == {"reward": 1.0, "verdict_label": "[[A=B]]"}
        server.server_client.post.assert_called_once()
        call_kwargs = server.server_client.post.call_args.kwargs
        assert call_kwargs["server_name"] == "equivalence_llm_judge"
        assert call_kwargs["url_path"] == "/verify"

    async def test_verify_raises_on_unknown_verifier_type(self) -> None:
        server = BraveSearchResourcesServer(config=self._config(), server_client=MagicMock(spec=ServerClient))
        request = self._verify_request(
            expected_answer="42", assistant_text="irrelevant", verifier_type="some_unconfigured_verifier"
        )

        with pytest.raises(ValueError, match="Unknown verifier"):
            await server.verify(request)

    async def test_search_recovers_after_mcp_session_reconnect(self) -> None:
        # The actual new behavior: a transient failure on the first call_tool() self-heals via
        # one reconnect+retry instead of failing the whole rollout turn.
        server = BraveSearchResourcesServer(config=self._config(), server_client=MagicMock(spec=ServerClient))
        bad_session = MagicMock()
        bad_session.call_tool = AsyncMock(side_effect=RuntimeError("transport died"))
        server._mcp_session = bad_session

        good_session = MagicMock()
        good_session.call_tool = AsyncMock(
            return_value=SimpleNamespace(
                isError=False,
                content=[SimpleNamespace(text="[1] Example (example.com)\n    URL: https://example.com\n")],
            )
        )

        async def fake_connect() -> None:
            server._mcp_session = good_session

        server._close_mcp_session = AsyncMock()
        server._connect_mcp_session = AsyncMock(side_effect=fake_connect)

        response = await server.search(BaseSearchQueryRequest(query="test query"))

        assert "example.com" in response.search_results
        bad_session.call_tool.assert_called_once()
        good_session.call_tool.assert_called_once()
        server._close_mcp_session.assert_awaited_once()
        server._connect_mcp_session.assert_awaited_once()
