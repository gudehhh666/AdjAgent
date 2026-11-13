"""Input cards bind named, authorized bytes, never directory guesses or answers."""
import hashlib
import json
import os

from adj_partner.input_binding import INPUT_GUIDANCE, task_input_cards


def config(root):
    return {'allowed_roots': [str(root)], 'allowed_tools': ['code_interpreter']}


def test_exact_csv_identity_and_sandbox_path(tmp_path):
    path = tmp_path / 'observations.csv'
    content = b'subject,value\na,3\nb,4\n'
    path.write_bytes(content)
    cards = task_input_cards(f'Count rows in `{path}`.', config(tmp_path), tmp_path)
    assert len(cards) == 1
    assert cards[0]['path'] == str(path)
    assert cards[0]['snapshot'] == 'inputs/observations.csv'
    assert cards[0]['sha256'] == hashlib.sha256(content).hexdigest()
    assert cards[0]['bytes'] == len(content)
    assert cards[0]['tools'] == ['code_interpreter']
    assert 'subject' not in str(cards)


def test_no_named_inputs_does_not_enumerate_authorized_directory(tmp_path):
    (tmp_path / 'private-answer.csv').write_text('answer,42')
    assert task_input_cards('Analyze the supplied table.', config(tmp_path), tmp_path) == []
    assert task_input_cards(str(tmp_path), config(tmp_path), tmp_path) == []


def test_authorization_and_tool_whitelist_are_preserved(tmp_path):
    allowed = tmp_path / 'inputs'
    allowed.mkdir()
    denied = tmp_path / 'outside.csv'
    denied.write_text('a,b')
    good = allowed / 'good.csv'
    good.write_text('c,d')
    assert task_input_cards(str(denied), config(allowed), tmp_path) == []
    cfg = {'allowed_roots': [str(allowed)], 'allowed_tools': []}
    assert task_input_cards(str(good), cfg, tmp_path) == []


def test_explicit_relative_file_with_spaces_and_duplicate_mentions(tmp_path):
    path = tmp_path / 'two words.csv'
    path.write_text('one\n2\n')
    cards = task_input_cards(f'Read [{path.name}]({path}) and `{path}`.',
                            config(tmp_path), tmp_path, explicit_files=[path.name])
    assert [c['path'] for c in cards] == [str(path)]


def test_symlinks_credentials_store_private_and_traversal_are_rejected(tmp_path):
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    target = tmp_path / 'outside.csv'
    target.write_text('no')
    (inputs / 'alias.csv').symlink_to(target)
    (inputs / 'aliasdir').symlink_to(tmp_path, target_is_directory=True)
    (inputs / '.env').write_text('secret')
    store = inputs / 'runtime'
    store.mkdir()
    (store / 'answer.csv').write_text('hidden')
    paths = [inputs / 'alias.csv', inputs / 'aliasdir/outside.csv', inputs / '.env',
             inputs / '../outside.csv', store / 'answer.csv']
    assert task_input_cards('', config(inputs), tmp_path, explicit_files=paths,
                            store_root=store) == []


def test_non_regular_files_never_block_and_same_basename_is_not_ambiguous(tmp_path):
    fifo = tmp_path / 'fifo.csv'
    os.mkfifo(fifo)
    a = tmp_path / 'a'
    b = tmp_path / 'b'
    a.mkdir()
    b.mkdir()
    (a / 'same.csv').write_text('a')
    (b / 'same.csv').write_text('b')
    cards = task_input_cards('', config(tmp_path), tmp_path,
                            explicit_files=[fifo, a / 'same.csv', b / 'same.csv'])
    assert len(cards) == 2
    assert all(c['separate_call_required'] for c in cards)


def test_guidance_separates_snapshot_and_computation():
    assert 'analysis_result.json' in INPUT_GUIDANCE and 'summary' in INPUT_GUIDANCE
    assert 'does not prove' in INPUT_GUIDANCE
    assert 'invent' in INPUT_GUIDANCE


def test_unquoted_relative_paths_and_single_quoted_space_paths(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'observations.csv').write_text('a\n1')
    (data / 'two words.csv').write_text('b\n2')
    cards = task_input_cards("Read data/observations.csv and 'data/two words.csv'.",
                            config(tmp_path), tmp_path)
    assert {c['path'] for c in cards} == {str(data / 'observations.csv'), str(data / 'two words.csv')}


def test_real_runner_json_prompt_preserves_nested_markdown_paths(tmp_path):
    from eval.adjuvant_reevaluation_pilot import prompt
    path = tmp_path / 'observations.csv'
    path.write_text('subject,value\na,2\n')
    case = {'id': 'generic', 'track': 'data', 'question': 'Count the rows.',
            'input_text': f'Read `{path}`\nsubject,value\na,2\n',
            'source_family': 'synthetic'}
    cards = task_input_cards(prompt(case), config(tmp_path), tmp_path)
    assert [c['path'] for c in cards] == [str(path)]


def test_json_public_text_excludes_private_answer_and_reference_values(tmp_path):
    public = tmp_path / 'input.csv'
    public.write_text('a\n1')
    hidden = tmp_path / 'answer.csv'
    hidden.write_text('count,1')
    text = 'Task: ' + json.dumps({'input_text': f'Read `{public}`\nData follow.',
                                 'private_notes': f'Read `{hidden}`',
                                 'reference': {'input_text': str(hidden)},
                                 'expected': str(hidden)})
    cards = task_input_cards(text, config(tmp_path), tmp_path)
    assert [c['path'] for c in cards] == [str(public)]
