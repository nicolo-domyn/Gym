from concurrent.futures.process import BrokenProcessPool
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from nemo_gym.config_types import ResourcesServerRef
from nemo_gym.server_utils import ServerClient
from resources_servers.brave_search.app import (
    BaseGetPageContentRequest,
    BaseSearchQueryRequest,
    BraveSearchResourcesServer,
    BraveSearchResourcesServerConfig,
    BraveSearchVerifyRequest,
    _normalize_option,
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


# [CUSTOM-BOX-PARSER 2026-11-27] / [CUSTOM-BROWSE-FIXES 2026-11-27]
def _minimal_pdf(text: str) -> bytes:
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = b"%PDF-1.4\n", []
    for i, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + obj + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return out


class TestBoxParser:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("First \\boxed{A}, but actually \\boxed{C}.", "C"),
            ("\\boxed{\\text{C}}", "\\text{C}"),
            ("\\boxed{\\frac{1}{2}}", "\\frac{1}{2}"),
            ("\\boxed {B}", "B"),
            ("\\boxed{C} and an unclosed \\boxed{D", "C"),
            (None, None),
        ],
    )
    def test_last_balanced_box(self, text, expected) -> None:
        assert box_parser(text) == expected

    @pytest.mark.parametrize(
        "option, expected",
        [
            ("\\text{C}", "C"),
            ("{{\\text{C}}}", "C"),
            ("{\\rm D}", "D"),
            ("\\textbf{ B }", "B"),
            (" A ", "A"),
            ("A, C", "A, C"),
            (None, None),
        ],
    )
    def test_normalize_option(self, option, expected) -> None:
        assert _normalize_option(option) == expected


class TestBrowse:
    @pytest.fixture
    def server(self):
        server = BraveSearchResourcesServer(config=TestApp()._config(), server_client=MagicMock(spec=ServerClient))
        yield server
        server._shutdown_extract_pool()

    def _mock_transport(self, monkeypatch, handler) -> None:
        real_client = httpx.AsyncClient

        def client_with_transport(*args, **kwargs):
            return real_client(*args, transport=httpx.MockTransport(handler), **kwargs)

        monkeypatch.setattr("resources_servers.brave_search.app.httpx.AsyncClient", client_with_transport)

    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost:8000",
            "http://127.0.0.1:8000/v1/models",
            "http://[::1]/",
            "http://10.1.0.2:8080/",
            "http://192.168.1.1/",
            "http://169.254.169.254/latest/meta-data/",
            "http://[::ffff:127.0.0.1]/",
            "file:///etc/passwd",
        ],
    )
    async def test_browse_blocks_non_public_urls(self, server, monkeypatch, url) -> None:
        handler = MagicMock(side_effect=AssertionError("request must not be sent"))
        self._mock_transport(monkeypatch, handler)
        response = await server.browse(BaseGetPageContentRequest(url=url))
        assert response.page_content.startswith("Error: URL not allowed")
        handler.assert_not_called()

    async def test_browse_blocks_redirect_to_loopback(self, server, monkeypatch) -> None:
        sent = []

        def handler(request: httpx.Request) -> httpx.Response:
            sent.append(str(request.url))
            return httpx.Response(302, headers={"Location": "http://127.0.0.1:8000/"})

        self._mock_transport(monkeypatch, handler)
        response = await server.browse(BaseGetPageContentRequest(url="http://93.184.215.14/"))
        assert response.page_content.startswith("Error: URL not allowed")
        assert sent == ["http://93.184.215.14/"]

    async def test_browse_sends_user_agent_and_reports_http_status(self, server, monkeypatch) -> None:
        seen = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.headers["user-agent"])
            return httpx.Response(404)

        self._mock_transport(monkeypatch, handler)
        response = await server.browse(BaseGetPageContentRequest(url="http://93.184.215.14/"))
        assert seen == [server.config.browse_user_agent]
        assert response.page_content == "Error: HTTP 404 Not Found"

    async def test_browse_retries_with_fallback_user_agent(self, server, monkeypatch) -> None:
        seen = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.headers["user-agent"])
            if request.headers["user-agent"] == server.config.browse_fallback_user_agent:
                return httpx.Response(200, text="served to browsers", headers={"content-type": "text/plain"})
            return httpx.Response(403)

        self._mock_transport(monkeypatch, handler)
        response = await server.browse(BaseGetPageContentRequest(url="http://93.184.215.14/"))
        assert seen == [server.config.browse_user_agent, server.config.browse_fallback_user_agent]
        assert response.page_content == "served to browsers"

    async def test_browse_fallback_disabled(self, server, monkeypatch) -> None:
        server.config.browse_fallback_user_agent = None
        handler = MagicMock(return_value=httpx.Response(403))
        self._mock_transport(monkeypatch, handler)
        response = await server.browse(BaseGetPageContentRequest(url="http://93.184.215.14/"))
        assert handler.call_count == 1
        assert response.page_content == "Error: HTTP 403 Forbidden"

    async def test_browse_accepts_any_2xx(self, server, monkeypatch) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(203, text="cached copy", headers={"content-type": "text/plain"})

        self._mock_transport(monkeypatch, handler)
        response = await server.browse(BaseGetPageContentRequest(url="http://93.184.215.14/"))
        assert response.page_content == "cached copy"

    async def test_browse_extracts_html(self, server, monkeypatch) -> None:
        body = "<html><body><article><p>" + "Acoustics paragraph. " * 20 + "</p></article></body></html>"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=body, headers={"content-type": "text/html; charset=utf-8"})

        self._mock_transport(monkeypatch, handler)
        response = await server.browse(BaseGetPageContentRequest(url="http://93.184.215.14/"))
        assert "Acoustics paragraph." in response.page_content

    async def test_browse_extraction_timeout(self, server, monkeypatch) -> None:
        server.config.browse_extract_timeout_sec = 1e-9

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=_minimal_pdf("slow"), headers={"content-type": "application/pdf"})

        self._mock_transport(monkeypatch, handler)
        response = await server.browse(BaseGetPageContentRequest(url="http://93.184.215.14/a.pdf"))
        assert response.page_content == "Error: could not extract the page text in time"

    async def test_browse_recreates_broken_extraction_pool(self, server, monkeypatch) -> None:
        broken = MagicMock()
        broken.submit.side_effect = BrokenProcessPool("worker died")
        server._extract_pool = broken

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="page", headers={"content-type": "text/plain"})

        self._mock_transport(monkeypatch, handler)
        response = await server.browse(BaseGetPageContentRequest(url="http://93.184.215.14/"))
        assert response.page_content == "Error: could not extract the page text"
        assert server._extract_pool is None
        broken.shutdown.assert_called_once()
        # The next call gets a fresh pool.
        response = await server.browse(BaseGetPageContentRequest(url="http://93.184.215.14/"))
        assert response.page_content == "page"

    async def test_browse_extracts_pdf_text(self, server, monkeypatch) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, content=_minimal_pdf("Hello from a PDF"), headers={"content-type": "application/pdf"}
            )

        self._mock_transport(monkeypatch, handler)
        response = await server.browse(BaseGetPageContentRequest(url="http://93.184.215.14/a.pdf"))
        assert "Hello from a PDF" in response.page_content

    async def test_browse_returns_plain_text_as_is(self, server, monkeypatch) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="Title: Page\n\nMarkdown body", headers={"content-type": "text/plain"})

        self._mock_transport(monkeypatch, handler)
        response = await server.browse(BaseGetPageContentRequest(url="http://93.184.215.14/"))
        assert response.page_content == "Title: Page\n\nMarkdown body"

    async def test_browse_timeout_message(self, server, monkeypatch) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timed out", request=request)

        self._mock_transport(monkeypatch, handler)
        response = await server.browse(BaseGetPageContentRequest(url="http://93.184.215.14/"))
        assert response.page_content == "Error: timed out after 10s"
