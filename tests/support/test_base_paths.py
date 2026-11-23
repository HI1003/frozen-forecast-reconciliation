"""Storage migration contracts; fixtures are kept in tmp (no recursive cleanup)."""

# Resolve the approved project layout while retaining sealed source-path keys.
from src.support.paths.project import project_file

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_file(ROOT)))
from src.support.paths.forecasts import base_forecast_path, manifest_base_path, resolve_base_path


class TestBasePaths(unittest.TestCase):
    def setUp(self):
        (project_file(ROOT/'tmp')).mkdir(exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix='base-path-contract-', dir=ROOT/'tmp'))
        self.old = self.root/'results/experiments/main/base_forecasts/walmart_chronos2.npz'
        self.new = base_forecast_path(self.root, 'main', 'walmart', 'chronos2')

    def test_main_is_dataset_only(self):
        self.assertEqual(self.new, self.root/'results/raw_local/base_forecasts/walmart/walmart_chronos2.npz')

    def test_historical_and_sidecar_resolve_to_canonical(self):
        self.assertEqual(resolve_base_path(self.old), self.new)
        self.assertEqual(resolve_base_path(self.old.with_suffix('.meta.json')), self.new.with_suffix('.meta.json'))

    def test_current_paths_are_idempotent(self):
        self.assertEqual(resolve_base_path(self.new), self.new)

    def test_experiments_remain_isolated(self):
        other = base_forecast_path(self.root, 'capacity-64x32', 'walmart', 'chronos2')
        self.assertNotEqual(other, self.new)
        self.assertEqual(other.parent, self.new.parent)
        old = self.root/'results/experiments/capacity-64x32/base_forecasts/walmart_chronos2.npz'
        self.assertEqual(resolve_base_path(old), other)

    def test_sensitivity_model_key_is_preserved(self):
        old = self.old.with_name('walmart_chronos2_synth.npz')
        self.assertEqual(resolve_base_path(old), base_forecast_path(self.root, 'main', 'walmart', 'chronos2_synth'))

    def test_old_exists_is_never_a_fallback(self):
        project_file(self.old.parent).mkdir(parents=True)
        project_file(self.old).write_bytes(b'old-only')
        with self.assertRaises(FileNotFoundError):
            project_file(resolve_base_path(self.old)).read_bytes()

    def test_legacy_smoke_is_not_main(self):
        old = self.root/'results/base_forecasts/walmart_timesfm25.npz'
        self.assertEqual(resolve_base_path(old), old)

    def test_manifest_keeps_original_identity_and_hash(self):
        project_file(self.new.parent).mkdir(parents=True)
        project_file(self.new).write_bytes(b'exact sealed bytes')
        expected = hashlib.sha256(project_file(self.new).read_bytes()).hexdigest()
        checksums = {self.old.relative_to(self.root).as_posix(): expected}
        logical = manifest_base_path(self.root, 'main', 'walmart', 'chronos2', checksums)
        self.assertEqual(logical, self.old)
        self.assertEqual(hashlib.sha256(project_file(resolve_base_path(logical)).read_bytes()).hexdigest(), expected)

    def test_new_manifest_uses_current_identity(self):
        checksums = {self.new.relative_to(self.root).as_posix(): 'a'*64}
        self.assertEqual(manifest_base_path(self.root, 'main', 'walmart', 'chronos2', checksums), self.new)

    def test_ambiguous_manifest_fails(self):
        checksums = {p.relative_to(self.root).as_posix(): 'a'*64 for p in (self.new, self.old)}
        with self.assertRaises(ValueError):
            manifest_base_path(self.root, 'main', 'walmart', 'chronos2', checksums)

    def test_unsafe_components_fail(self):
        for experiment, dataset, base in [('..', 'walmart', 'chronos2'), ('main', '../walmart', 'chronos2'), ('main', 'walmart', '../chronos2')]:
            with self.assertRaises(ValueError):
                base_forecast_path(self.root, experiment, dataset, base)


if __name__ == '__main__':
    unittest.main()
