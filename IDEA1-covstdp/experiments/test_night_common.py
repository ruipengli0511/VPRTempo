"""MIT licensed common-epoch selection tests independent of query scoring."""
import copy
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_night_common import choose_common


def main():
    rows = []
    for seed in range(3):
        rows.append({'status': 'complete', 'phase': 'night_finetune', 'method': 'anchor', 'seed': seed,
            'arm': 'retention', 'retention_coefficient': 100., 'calibration': {'lr': .1},
            'probe_identity': {'tensor_sha256': 'same'}, 'epoch_results': [
              {'epoch': 0, 'development': {'R@1': .8, 'R@5': .99}, 'eligible_for_selection': True},
              {'epoch': 1, 'development': {'R@1': [.9,.8,.8][seed], 'R@5': .5}, 'eligible_for_selection': True},
              {'epoch': 2, 'development': {'R@1': [.8,.9,.8][seed], 'R@5': 1.}, 'eligible_for_selection': True}]})
    assert choose_common(rows)[0]['epoch'] == 1, 'sum R1 tie must select earliest, not R5'
    for changed in [rows[:2], rows + [copy.deepcopy(rows[0])]]:
        try: choose_common(changed)
        except ValueError: pass
        else: raise AssertionError('missing or duplicate seeds accepted')
    bad = copy.deepcopy(rows); bad[1]['retention_coefficient'] = 30.
    try: choose_common(bad)
    except ValueError: pass
    else: raise AssertionError('different candidate accepted')
    partial = copy.deepcopy(rows); partial[2]['epoch_results'][1]['eligible_for_selection'] = False
    assert choose_common(partial)[0]['epoch'] == 2
    print('PASS: shared epoch from R1 sums, earliest tie, R5 excluded, missing/duplicate seed rejection, coefficient identity, incomplete epoch exclusion')


if __name__ == '__main__': main()
