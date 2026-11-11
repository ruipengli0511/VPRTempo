"""Operational tests only: immutable snapshots, paths and environment identity."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from vprtempo.src.run_provenance import code_identity, environment_identity, resolve_locations, runtime_identity


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def make_snapshot(root):
    root.mkdir()
    files = {'pixi.lock': b'locked\n', 'main.py': b'pass\n'}
    for name, data in files.items():
        (root / name).write_bytes(data)
    manifest = {'project': 'VPRTempo', 'source_commit': 'a' * 40, 'source_tree': 'b' * 40,
                'deployment_id': 'full-' + 'a' * 12, 'target_dir': str(root),
                'lock_sha256': hashlib.sha256(files['pixi.lock']).hexdigest(),
                'files': [{'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                          for name, data in files.items()]}
    data = encoded(manifest)
    digest = hashlib.sha256(data).hexdigest()
    (root / 'deployment_manifest.json').write_bytes(data)
    (root / 'deployment_receipt.json').write_bytes(encoded({
        'manifest_sha256': digest, 'source_commit': 'a' * 40, 'target_dir': str(root),
        'deployed_at_utc': '2026-11-11T00:00:00+00:00'}))
    return digest


class OperationalAcceptance(unittest.TestCase):
    def test_snapshot_without_git_and_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'snapshot'
            digest = make_snapshot(root)
            with patch('subprocess.check_output', side_effect=AssertionError('must not call Git')):
                identity = code_identity(root, expected_manifest_sha=digest)
                self.assertEqual(identity['source'], 'verified_deployment_snapshot')
            with self.assertRaises(ValueError):
                code_identity(root)
            with self.assertRaises(ValueError):
                code_identity(root, expected_manifest_sha='0' * 64)
            (root / 'main.py').write_text('modified')
            with self.assertRaises(ValueError):
                code_identity(root, expected_manifest_sha=digest)

    def test_explicit_paths_and_precreated_slurm_parent(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            root, data = temp / 'snapshot', temp / 'raw'
            cfg = {'data_dir': str(data), 'run_root': str(temp / 'Runs'),
                   'cache_dir': str(temp / 'Cache'), 'run_id': 'test-001'}
            artifacts, cache = resolve_locations(cfg, root, deployed=True)
            artifacts.parent.mkdir(parents=True)
            (artifacts.parent / 'slurm-123.out').write_text('log')
            artifacts.mkdir()
            with self.assertRaises(FileExistsError):
                artifacts.mkdir()
            self.assertEqual(cache, temp / 'Cache')
            for bad in ('../escape', 'path/name', '/absolute', '..'):
                with self.assertRaises(ValueError):
                    resolve_locations({**cfg, 'run_id': bad}, root, deployed=True)
            with self.assertRaises(ValueError):
                resolve_locations({**cfg, 'cache_dir': str(data / 'cache')}, root, deployed=True)
            with self.assertRaises(ValueError):
                resolve_locations({**cfg, 'run_root': str(root / 'runs')}, root, deployed=True)

    def test_environment_prefix_and_package_fingerprint(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            prefix = temp / 'env'
            records = prefix / 'conda-meta'
            records.mkdir(parents=True)
            package = {'name': 'python', 'version': '3.12.13', 'build': 'example',
                       'build_number': 0, 'sha256': None, 'md5': None}
            (records / 'python.json').write_bytes(encoded(package))
            lock = temp / 'pixi.lock'
            lock.write_text('lock')
            lock_sha = hashlib.sha256(lock.read_bytes()).hexdigest()
            fingerprint = hashlib.sha256(encoded({'schema_version': 1,
                         'lock_sha256': lock_sha, 'packages': [package]})).hexdigest()
            manifest = temp / 'environment.json'
            manifest.write_bytes(encoded({'prefix': str(prefix), 'lock_sha256': lock_sha,
                'package_record_fingerprint': fingerprint, 'environment_id': 'pixi-cuda-' + fingerprint[:16]}))
            with patch.object(sys, 'prefix', str(prefix)):
                self.assertEqual(environment_identity(manifest, lock)['package_count'], 1)
                package['version'] = '3.12.14'
                (records / 'python.json').write_bytes(encoded(package))
                with self.assertRaises(ValueError):
                    environment_identity(manifest, lock)
            with self.assertRaises(ValueError):
                environment_identity(manifest, lock)

    def test_runtime_metadata_loads_with_weights_only(self):
        import torch
        import numpy
        with tempfile.TemporaryDirectory() as temp:
            record = runtime_identity(torch, numpy)
            self.assertIs(type(record['torch']), str)
            path = Path(temp) / 'checkpoint.pth'
            torch.save({'weight': torch.ones(2), 'runtime': record}, path)
            loaded = torch.load(path, weights_only=True)
            self.assertEqual(loaded['runtime']['torch'], str(torch.__version__))


if __name__ == '__main__':
    unittest.main(verbosity=2)
