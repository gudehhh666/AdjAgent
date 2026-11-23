import pytest

from adj_partner.contracts import ModelReply
from adj_partner.models import base
from adj_partner.models.base import ProviderFailure, ScriptedPort
from adj_partner.models.requests import public_request


MESSAGES = [{'role': 'user', 'content': 'Analyze the public inputs.'}]
TOOLS = [{'type': 'function', 'function': {'name': 'analyze', 'parameters': {
    'type': 'object', 'properties': {}, 'required': [], 'additionalProperties': False}}}]


def test_default_request_retains_existing_shape():
    expected = {'messages': MESSAGES, 'tools': TOOLS, 'max_output_tokens': 128, 'timeout_seconds': 10}
    assert public_request(MESSAGES, TOOLS, 128, 10) == expected
    assert public_request(MESSAGES, TOOLS, 128, 10, tool_choice='auto') == expected


def test_required_action_is_explicit_and_snapshot_is_copied():
    request = public_request(MESSAGES, TOOLS, 128, 10, tool_choice='required_or_refuse')
    assert request['tool_choice'] == 'required_or_refuse'
    request['tools'].clear()
    assert len(TOOLS) == 1


@pytest.mark.parametrize('choice,tools', [('required_or_refuse', []), ('required', TOOLS),
                                        (True, TOOLS), (None, TOOLS), ([], TOOLS)])
def test_invalid_or_unsatisfiable_modes_fail_before_dispatch(choice, tools):
    with pytest.raises(ProviderFailure) as exc:
        public_request(MESSAGES, tools, 128, 10, tool_choice=choice)
    assert exc.value.code == 'invalid_request' and not exc.value.cost_unknown


def test_capability_requires_explicit_typed_declaration():
    check = getattr(base, 'supports_tool_choice', None)
    assert callable(check), 'typed tool-choice capability is not implemented'
    assert check(object(), 'auto')
    assert not check(object(), 'required_or_refuse')
    for modes, expected in [(['auto','required_or_refuse'], True), (('required_or_refuse',), True),
                            ('required_or_refuse', False), (True, False), (['auto'], False)]:
        port = type('Port', (), {'metadata': {'tool_choice_modes': modes}})()
        assert check(port, 'required_or_refuse') is expected
        assert not check(port, 'required')


def test_scripted_port_preserves_required_mode_and_explicit_refusal():
    port = ScriptedPort([ModelReply(refusal='Cannot execute the requested analysis.',
                                   usage={'input_tokens': 3, 'output_tokens': 2})])
    reply = port.complete(MESSAGES, TOOLS, max_output_tokens=128, timeout_seconds=10,
                          tool_choice='required_or_refuse')
    assert port.metadata['tool_choice_modes'] == ['auto', 'required_or_refuse']
    assert port.requests[0]['tool_choice'] == 'required_or_refuse'
    assert reply.refusal and not reply.tool_calls
    assert reply.usage == {'input_tokens': 3, 'output_tokens': 2}
