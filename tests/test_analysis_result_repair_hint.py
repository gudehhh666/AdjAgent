import json
import pytest
from adj_partner.analysis.worker import _validated_result
from adj_partner.tools import ToolFailure


@pytest.mark.parametrize('payload', [{'count': 1}, {'summary': {}, 'method': 'read file'},
                                     {'summary': {}, 'limitations': 'none'}])
def test_invalid_result_explains_complete_contract_without_weakening_validation(payload):
    with pytest.raises(ToolFailure) as error:
        _validated_result(json.dumps(payload).encode())
    assert error.value.code == 'analysis_output_invalid'
    message = str(error.value)
    assert 'summary' in message and 'metrics' in message and 'limitations' in message
    assert 'array of strings' in message
    assert 'final answer' in message


def test_repaired_result_uses_existing_schema():
    value = {'summary': {'count': 1, 'method': 'read file'}, 'limitations': ['metadata only']}
    assert _validated_result(json.dumps(value).encode()) == value
