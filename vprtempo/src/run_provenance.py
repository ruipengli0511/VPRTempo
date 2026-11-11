"""Operational provenance/path adapters; no feature or learning logic."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import subprocess
import sys


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def code_identity(root, manifest_path=None, expected_manifest_sha=None):
    root = Path(root).resolve()
    manifest_path = Path(manifest_path).resolve() if manifest_path else root / 'deployment_manifest.json'
    if manifest_path.exists() or expected_manifest_sha is not None:
        if manifest_path != root / 'deployment_manifest.json' or not expected_manifest_sha:
            raise ValueError('snapshot requires its root manifest and an independent expected SHA256')
        if root.is_symlink() or manifest_path.is_symlink() or file_sha(manifest_path) != expected_manifest_sha:
            raise ValueError('deployment manifest integrity check failed')
        actual = set()
        for parent, dirs, files in os.walk(root, followlinks=False):
            if any((Path(parent) / name).is_symlink() for name in dirs + files):
                raise ValueError('symlink in deployment snapshot')
            actual.update((Path(parent) / name).relative_to(root).as_posix() for name in files)
        manifest = json.loads(manifest_path.read_bytes())
        expected = {'deployment_manifest.json', 'deployment_receipt.json'}
        if manifest['project'] != 'VPRTempo' or not re.fullmatch('[0-9a-f]{40}', manifest['source_commit']):
            raise ValueError('invalid deployment source identity')
        for entry in manifest['files']:
            name = entry['path']
            relative = PurePosixPath(name)
            if (relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name
                    or str(relative) != name or name in expected):
                raise ValueError('unsafe/duplicate source manifest path')
            expected.add(name)
            path = root / name
            if not path.is_file() or path.stat().st_size != entry['bytes'] or file_sha(path) != entry['sha256']:
                raise ValueError('snapshot source changed: ' + name)
        if actual != expected:
            raise ValueError('unexpected or missing snapshot files: ' + str(sorted(actual ^ expected)))
        receipt_path = root / 'deployment_receipt.json'
        receipt = json.loads(receipt_path.read_bytes())
        if (receipt['manifest_sha256'] != expected_manifest_sha
                or receipt['source_commit'] != manifest['source_commit']
                or receipt['target_dir'] != str(root) or manifest['target_dir'] != str(root)):
            raise ValueError('publication receipt does not match this snapshot')
        if file_sha(root / 'pixi.lock') != manifest['lock_sha256']:
            raise ValueError('snapshot lock identity mismatch')
        return {'commit': manifest['source_commit'], 'tree': manifest['source_tree'],
                'branch': None, 'dirty': False, 'source': 'verified_deployment_snapshot',
                'deployment_id': manifest['deployment_id'], 'manifest_sha256': expected_manifest_sha,
                'receipt_sha256': file_sha(receipt_path), 'deployed_at_utc': receipt['deployed_at_utc']}
    if manifest_path != root / 'deployment_manifest.json':
        raise FileNotFoundError(manifest_path)
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    return {'commit': git('rev-parse', 'HEAD'), 'branch': git('branch', '--show-current'),
            'dirty': bool(git('status', '--porcelain=v1', '--untracked-files=all')),
            'source': 'local_git'}


def environment_identity(manifest_path, lock_path, required=False):
    if not manifest_path:
        if required:
            raise ValueError('published snapshot requires an environment manifest')
        return {'source': 'local_runtime', 'prefix': sys.prefix}
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_bytes())
    prefix = Path(manifest['prefix']).resolve()
    if Path(sys.prefix).resolve() != prefix:
        raise ValueError('Python is not running in the approved environment prefix')
    lock_sha = file_sha(lock_path)
    if lock_sha != manifest['lock_sha256']:
        raise ValueError('source lock differs from the approved environment lock')
    records = sorted((prefix / 'conda-meta').glob('*.json'))
    if not records:
        raise ValueError('environment package records missing')
    packages = []
    for record_path in records:
        record = json.loads(record_path.read_bytes())
        packages.append({key: record.get(key) for key in
                         ('name', 'version', 'build', 'build_number', 'sha256', 'md5')})
    packages.sort(key=lambda p: (p['name'], p['version'], p['build']))
    identity = {'schema_version': 1, 'lock_sha256': lock_sha, 'packages': packages}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True,
                                            separators=(',', ':')).encode()).hexdigest()
    if fingerprint != manifest['package_record_fingerprint']:
        raise ValueError('installed environment package records changed')
    if manifest['environment_id'] != 'pixi-cuda-' + fingerprint[:16]:
        raise ValueError('environment label/fingerprint mismatch')
    return {'source': 'verified_conda_records', 'environment_id': manifest['environment_id'],
            'manifest_path': str(path), 'manifest_sha256': file_sha(path),
            'package_record_fingerprint': fingerprint, 'package_count': len(packages),
            'prefix': str(prefix), 'lock_sha256': lock_sha,
            'fingerprint_scope': 'lock and installed package records, not all library bytes'}


def resolve_locations(cfg, root, deployed=False):
    root = Path(root).resolve()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,119}', cfg['run_id']):
        raise ValueError('run_id must be a single safe directory name (up to 120 characters)')
    if deployed and any(not cfg.get(key) for key in ('data_dir', 'cache_dir', 'run_root')):
        raise ValueError('published run requires explicit data_dir, cache_dir and run_root')
    data = Path(cfg['data_dir']).expanduser().resolve()
    runs = Path(cfg.get('run_root') or root / 'runs/task_feature').expanduser().resolve()
    cache = Path(cfg.get('cache_dir') or runs / 'cache').expanduser().resolve()
    for path in (runs, cache):
        if path == data or data in path.parents:
            raise ValueError('outputs/cache must not be written in the raw data tree')
        if deployed and (path == root or root in path.parents):
            raise ValueError('published source snapshot must remain read-only')
    run_dir = runs / cfg['run_id']
    if cache == run_dir or run_dir in cache.parents:
        raise ValueError('shared input cache cannot be stored inside this run')
    cfg.update({'data_dir': str(data), 'run_root': str(runs), 'cache_dir': str(cache),
                'run_dir': str(run_dir), 'artifact_dir': str(run_dir / 'artifacts')})
    return run_dir / 'artifacts', cache


def runtime_identity(torch_module, numpy_module):
    return {'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
            'hostname': platform.node(), 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
            'python': platform.python_version(), 'executable': sys.executable,
            'torch': str(torch_module.__version__), 'torch_cuda': torch_module.version.cuda,
            'numpy': str(numpy_module.__version__)}
