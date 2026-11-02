# [CUSTOM] Markdown code fence policy: optional by default, forbidden when the prompt says so, required on demand.
import json
from unittest.mock import MagicMock

import pytest

from nemo_gym.server_utils import ServerClient
from resources_servers.structured_outputs.app import (
    SchemaType,
    StructuredOutputsResourcesServer,
    StructuredOutputsResourcesServerConfig,
    prompt_forbids_code_fence,
    split_code_fence,
)


SCHEMA = json.dumps({"type": "object", "properties": {"name": {"type": "string"}}})
JSON_ANSWER = '{"name": "Ada"}'
YAML_ANSWER = "name: Ada"


def _score(schema_type, text, code_block):
    config = StructuredOutputsResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
    server = StructuredOutputsResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    return server.evaluate_structured_output_response(schema_type, SCHEMA, text, code_block=code_block)[:2]


@pytest.mark.parametrize(
    "schema_type,text,code_block,expected",
    [
        (SchemaType.JSON, JSON_ANSWER, "optional", (1.0, None)),
        (SchemaType.JSON, f"```json\n{JSON_ANSWER}\n```", "optional", (1.0, None)),
        (SchemaType.JSON, f"```\n{JSON_ANSWER}\n```", "required", (1.0, None)),
        (SchemaType.YAML, f"```yml\n{YAML_ANSWER}\n```", "required", (1.0, None)),
        (SchemaType.JSON, JSON_ANSWER, "required", (0.0, "code_block_missing")),
        (SchemaType.JSON, f"```json\n{JSON_ANSWER}\n```", "forbidden", (0.0, "code_block_forbidden")),
        (SchemaType.YAML, f"```json\n{YAML_ANSWER}\n```", "optional", (0.0, "code_block_language")),
        (SchemaType.JSON, f"Here it is:\n```json\n{JSON_ANSWER}\n```", "optional", (0.0, "parse_error")),
    ],
)
def test_code_fence_policy(schema_type, text, code_block, expected):
    assert _score(schema_type, text, code_block) == expected


def test_split_code_fence():
    assert split_code_fence("```YAML\na: 1\n```") == ("yaml", "a: 1")
    assert split_code_fence("a: 1") is None
    assert split_code_fence("```yaml\na: 1\n``` trailing") is None


def test_prompt_forbids_code_fence():
    def params(text):
        return {"input": [{"role": "user", "content": text}]}

    assert prompt_forbids_code_fence(params("Return only the YAML output without styling it in backticks."))
    assert prompt_forbids_code_fence(params("Answer, not in ```json``` format."))
    assert prompt_forbids_code_fence(params("Return raw TOML without markdown fencing."))
    assert not prompt_forbids_code_fence(params("Format your response as a YAML document."))
