"""MIT licensed acceptance checks for sparse queries and shared-cache safety."""
import multiprocessing as mp
from pathlib import Path
import random
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from vprtempo.src.task_feature import _rank, schedule
from vprtempo.src.task_protocol import cached_tensor, query_ids_from_config


def check_queries():
    db = torch.eye(3).repeat(2, 1)
    q = torch.eye(3)[[2, 0]]
    result = _rank(db, q, 3, [2, 0])
    assert result['place_R@1'] == 1 and result['n_queries'] == 2
    assert [p['gt'] for p in result['predictions']] == [2, 0]
    assert query_ids_from_config({'n_places': 3300, 'query_start': 600, 'query_count': 500}) == list(range(600, 1100))
    for cfg in ({'query_ids': [1, 1]}, {'query_ids': [3]}, {'query_ids': []}, {'query_role': 'retained'}):
        try:
            query_ids_from_config({'n_places': 3, **cfg})
        except ValueError:
            pass
        else:
            raise AssertionError(f'invalid query configuration accepted: {cfg}')


def check_cache():
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        cache, marker = folder / 'cache.pt', folder / 'built.txt'
        def worker():
            def build():
                with marker.open('a') as stream:
                    stream.write('one\n')
                return {'tensor': torch.arange(7)}
            assert torch.equal(cached_tensor(cache, build)['tensor'], torch.arange(7))
        ctx = mp.get_context('fork')
        processes = [ctx.Process(target=worker) for _ in range(3)]
        for process in processes:
            process.start()
        for process in processes:
            process.join(30)
            if process.is_alive():
                process.terminate()
                raise AssertionError('cache lock deadlock')
            assert process.exitcode == 0
        assert marker.read_text() == 'one\n'
        assert not list(folder.glob('*.tmp'))
        try:
            cached_tensor(folder / 'missing.pt', lambda: {}, read_only=True)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError('read-only mode generated a missing input cache')


def check_sampler_equivalence():
    # Reference the previous list-based sampler to ensure the O(1) mapping is exact.
    torch.manual_seed(31)
    initial = {s: torch.nn.functional.normalize(torch.rand(100, 12), dim=1) for s in ('spring', 'fall')}
    rng = random.Random(5)
    expected = []
    for _ in range(2):
        anchors = [(s, p) for s in (0, 1) for p in range(100)]
        rng.shuffle(anchors)
        epoch = []
        for index, (s, p) in enumerate(anchors):
            legal = [n for n in range(100) if abs(n-p) >= 20]
            if index % 2 == 0:
                negative = rng.choice(legal)
            else:
                candidates = rng.sample(legal, min(16, len(legal)))
                other = 'fall' if s == 0 else 'spring'
                anchor = initial['spring' if s == 0 else 'fall'][p]
                negative = max(candidates, key=lambda n: (float(anchor @ initial[other][n]), -n))
            epoch.append((s, p, negative))
        expected.append(epoch)
    assert schedule(100, 2, 5, initial, 20)[0] == expected
    scale = {s: torch.nn.functional.normalize(torch.rand(3300, 12), dim=1)
             for s in ('spring', 'fall')}
    large, _ = schedule(3300, 1, 0, scale, 50)
    assert len(large[0]) == 6600
    assert len({(s, a) for s, a, _ in large[0]}) == 6600
    assert all(abs(a-n) >= 50 for _, a, n in large[0])


if __name__ == '__main__':
    check_queries()
    check_cache()
    check_sampler_equivalence()
    from summarize_task_batch import select_groups
    rows = [{'task': {'method': 'full', 'frontend': 'gray', 'lr': 0.001, 'seed': seed},
             'summary': {'epoch_results': [{'epoch': 1, 'place_R@1': a},
                                           {'epoch': 4, 'place_R@1': b}]}}
            for seed, (a, b) in enumerate(((0.9, 0.3), (0.3, 0.9), (0.4, 0.4)))]
    winner = select_groups(rows)['winners'][0]
    assert winner['epoch'] == 1 and abs(winner['mean_R@1'] - 1.6/3) < 1e-12
    print('query IDs, retained-query guard, concurrent cache and sampler equivalence passed')
