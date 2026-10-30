# [CUSTOM] Arguments beyond the label pass only when they repeat the tool schema's default (BFCL rule).
import json
from unittest.mock import MagicMock

import pytest

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from app import XlamFcResourcesServer, XlamFcResourcesServerConfig, XlamFcVerifyRequest


TOOL = {
    "type": "function",
    "name": "sticker_search",
    "strict": False,
    "parameters": {
        "type": "object",
        "properties": {
            "q": {"type": "string"},
            "limit": {"type": "integer", "default": 25},
            "rating": {"type": "string", "default": "G"},
            "lang": {"type": "string"},
        },
        "required": ["q"],
    },
}
EXPECTED = [{"name": "sticker_search", "arguments": {"q": "gaming"}}]


async def _reward(arguments, tools=(TOOL,)):
    response = NeMoGymResponse(
        id="resp",
        created_at=0.0,
        model="dummy",
        object="response",
        output=[
            {"call_id": "c1", "name": "sticker_search", "arguments": json.dumps(arguments), "type": "function_call"}
        ],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )
    server = XlamFcResourcesServer(
        config=XlamFcResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name=""),
        server_client=MagicMock(spec=ServerClient),
    )
    request = XlamFcVerifyRequest(
        responses_create_params={
            "input": [{"role": "user", "content": "Find gaming stickers."}],
            "tools": list(tools),
        },
        response=response,
        expected_answers=EXPECTED,
    )
    return (await server.verify(request)).reward


@pytest.mark.parametrize(
    "arguments,reward",
    [
        ({"q": "gaming"}, 1.0),
        ({"q": "gaming", "limit": 25}, 1.0),  # repeats the default
        ({"q": "gaming", "limit": 25.0, "rating": "g"}, 1.0),  # same value, BFCL-style comparison
        ({"q": "gaming", "limit": 10}, 0.0),  # invented non-default value
        ({"q": "gaming", "lang": "en"}, 0.0),  # optional, no default
        ({"q": "gaming", "page": 1}, 0.0),  # not in the schema
        ({"q": "Gaming"}, 0.0),  # labelled arguments still match exactly
    ],
)
async def test_extra_arguments(arguments, reward):
    assert await _reward(arguments) == reward
