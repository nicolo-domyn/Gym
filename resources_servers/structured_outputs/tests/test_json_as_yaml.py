# [CUSTOM] A YAML request is not satisfied by JSON (flow-style) output; block YAML still passes.
import json
from unittest.mock import MagicMock

from nemo_gym.server_utils import ServerClient
from resources_servers.structured_outputs.app import (
    SchemaType,
    StructuredOutputsResourcesServer,
    StructuredOutputsResourcesServerConfig,
    is_flow_style_yaml,
)


SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "meta": {"type": "object", "properties": {}},
        },
    }
)


def _score(text: str):
    config = StructuredOutputsResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
    server = StructuredOutputsResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    return server.evaluate_structured_output_response(SchemaType.YAML, SCHEMA, text)


def test_block_yaml_passes_with_flow_scalar_sequence_and_empty_mapping():
    assert _score("name: Ada\ntags: [math, code]\nmeta: {}\n") == (1.0, None, None)


def test_json_answer_to_yaml_request_fails():
    reward, error_type, _ = _score('{"name": "Ada", "tags": ["math"], "meta": {}}')
    assert (reward, error_type) == (0.0, "json_as_yaml")


def test_flow_mapping_inside_block_yaml_is_flagged():
    assert is_flow_style_yaml("name: Ada\nmeta:\n  inner: {a: 1}\n")
    assert not is_flow_style_yaml("name: Ada\ntags: [a, b]\nmeta: {}\n")
