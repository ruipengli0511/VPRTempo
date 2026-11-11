"""MIT licensed batch analysis: select shared lr/epoch over all three seeds."""
import argparse
import json
from pathlib import Path
import statistics


def select_groups(rows):
    grouped = {}
    for row in rows:
        task, summary = row['task'], row['summary']
        key = (task['method'], task['frontend'], task['lr'])
        grouped.setdefault(key, []).append(row)
    candidates = []
    for (method, frontend, lr), group in grouped.items():
        if sorted(r['task']['seed'] for r in group) != [0, 1, 2]:
            raise ValueError('each configuration requires exactly seeds 0,1,2')
        metrics = [{e['epoch']: e for e in r['summary']['epoch_results']
                    if not e.get('partial_epoch', False)} for r in group]
        epochs = sorted(set.intersection(*(set(m) for m in metrics)))
        for epoch in epochs:
            values = [m[epoch]['place_R@1'] for m in metrics]
            candidates.append({'method': method, 'frontend': frontend, 'lr': lr,
                               'epoch': epoch, 'seed_R@1': values,
                               'mean_R@1': statistics.mean(values),
                               'std_R@1': statistics.stdev(values)})
    winners = []
    for method, frontend in sorted({(c['method'], c['frontend']) for c in candidates}):
        eligible = [c for c in candidates if c['method'] == method and c['frontend'] == frontend]
        winners.append(min(eligible, key=lambda c: (-c['mean_R@1'], c['lr'], c['epoch'])))
    return {'candidates': candidates, 'winners': winners}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request', type=Path, required=True)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    request = json.loads(args.request.read_text())
    rows, missing = [], []
    pair_identities = {}
    for task in request['tasks']:
        folder = args.run_root / task['run_id'] / 'artifacts'
        if not (folder / 'summary.json').exists():
            missing.append(task['run_id'])
            continue
        summary = json.loads((folder / 'summary.json').read_text())
        cfg = json.loads((folder / 'effective_config.json').read_text())['config']
        ident = json.loads((folder / 'run_identity.json').read_text())
        if summary['status'] != 'complete':
            missing.append(task['run_id'])
            continue
        expected_commit = request['run_commits'][task['method']]
        code = ident.get('code', ident['git'])
        if code['commit'] != expected_commit or code.get('dirty'):
            raise ValueError(f'wrong source identity: {task["run_id"]}')
        expected_method = 'triplet_' + task['method']
        if (cfg['n_places'] != 3300 or cfg['query_ids'] != list(range(600,1100))
                or cfg['query_role'] != 'development' or cfg['method'] != expected_method
                or cfg['frontend'] != task['frontend'] or cfg['seed'] != task['seed']
                or cfg['lr'] != task['lr'] or cfg['epochs'] != 20
                or cfg['negative_min_gap'] != 50 or cfg['batch_size'] != 16
                or cfg['margin'] != 0.1 or cfg['feature_width'] != 2304
                or (task['frontend'] == 'gabor' and cfg['gabor_block'] != 4)):
            raise ValueError(f'wrong effective protocol: {task["run_id"]}')
        if [e['epoch'] for e in summary['epoch_results']] != [0,1,4,10,20]:
            raise ValueError('missing registered evaluation epochs')
        key = (task['frontend'], task['seed'])
        pairing = tuple(ident[k] for k in ('initial_weight_sha256', 'schedule_sha256',
                        'data_manifest_sha256', 'gabor_state_sha256'))
        if key in pair_identities and pair_identities[key] != pairing:
            raise ValueError(f'unpaired input/initialization/sampler: {task["run_id"]}')
        pair_identities[key] = pairing
        rows.append({'task': task, 'summary': summary})
    result = {'status': 'incomplete' if missing else 'complete', 'missing': missing,
              'completed': len(rows), 'query_role': 'development'}
    if not missing:
        result.update(select_groups(rows))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'completed={len(rows)} missing={len(missing)} output={args.output}')
    if missing:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
