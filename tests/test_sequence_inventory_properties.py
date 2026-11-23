"""Constructive metamorphic checks with independently generated record counts."""
import importlib
import importlib.util
import random
import zipfile

import pytest

from adj_partner.contracts import Budget, SessionConfig, ToolCall
from adj_partner.storage import Store
from adj_partner.tools import ToolCatalog, ToolExecutor


@pytest.mark.parametrize('seed', [19, 103, 9001])
def test_record_totals_are_permutation_and_batch_invariant(tmp_path, seed):
    assert importlib.util.find_spec('adj_partner.native.sequence_inventory'), 'inventory required'
    mod = importlib.import_module('adj_partner.native.sequence_inventory')
    rng = random.Random(seed)
    paths, expected = [], dict(files=0, records=0, residues=0, gaps=0, stops=0)
    for archive_index in range(3):
        p = tmp_path / f'source{archive_index}.zip'
        paths.append(str(p))
        with zipfile.ZipFile(p, 'w') as z:
            for category in ('keep', 'exclude'):
                for stem in ('a', 'b', 'c'):
                    records = []
                    n = rng.randint(1, 4)
                    for i in range(n):
                        residues = rng.randint(1, 30)
                        gaps, stops = rng.randint(0, 3), rng.randint(0, 2)
                        records.append('>repeated_id description\n' + 'A' * residues + '-' * gaps + '*' * stops + '\n')
                        if category == 'keep' and stem in {'a', 'c'}:
                            for k, v in [('records', 1), ('residues', residues), ('gaps', gaps), ('stops', stops)]:
                                expected[k] += v
                    z.writestr(f'{category}/{stem}.faa', ''.join(records))
                    if category == 'keep' and stem in {'a', 'c'}:
                        expected['files'] += 1
    store = Store(tmp_path / 'state')
    cat = ToolCatalog(); mod.register_sequence_inventory(cat)
    sid = store.create_session(SessionConfig(allowed_roots=[str(tmp_path)], allowed_tools=cat.names(), enabled_tools=cat.names()).to_dict())['id']
    task = store.create_task(sid, 'constructive counts')['id']
    rid = store.create_run(sid, task, Budget().to_dict())['id']
    executor = ToolExecutor(store, cat)
    def count(group, key):
        r = executor.execute(ToolCall(key, 'sequence_inventory', dict(paths=group, member_glob='keep/*', file_stems=['a', 'c'], scope_note='Synthetic explicit subset')), sid, rid)
        assert r.status == 'ok', r.to_dict()
        return r.data['counts']['after_stem_filter']
    assert count(paths, 'together') == expected
    assert count(list(reversed(paths)), 'reversed') == expected
    batched = [count([p], f'batch{i}') for i, p in enumerate(paths)]
    assert {k: sum(v[k] for v in batched) for k in expected} == expected
    store.close()
