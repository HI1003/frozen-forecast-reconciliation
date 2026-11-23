"""Registry invariants for the frozen base models (numpy-free).

The registry is the single place that decides which checkpoint a base key means
and where it may be run.  Both are load-bearing: a wrong model ID mislabels the
frozen forecasts, and running the synthetic-only sensitivity arm outside Walmart
would produce a table that looks like evidence about GEFCom contamination when
it is nothing of the kind.
"""

# Resolve the approved project layout while retaining sealed source-path keys.
from src.support.paths.project import project_file

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_file(ROOT)))

from src.base_forecasts.registry import BASE_MODELS, MAIN_BASES, SENSITIVITY_BASES, check_base_dataset, get_base


class TestRegistry(unittest.TestCase):
    def test_main_revisions_are_exact_frozen_commits(self):
        self.assertEqual(
            {key: BASE_MODELS[key].revision for key in MAIN_BASES},
            {
                "chronos2": "29ec3766d36d6f73f0696f85560a422f50e8498c",
                "timesfm25": "1d952420fba87f3c6dee4f240de0f1a0fbc790e3",
            },
        )
        self.assertEqual(len({BASE_MODELS[key].revision for key in MAIN_BASES}), 2)

    def test_keys_match_their_spec(self):
        for key, spec in BASE_MODELS.items():
            self.assertEqual(key, spec.key)

    def test_model_ids_are_unique(self):
        ids = [s.model_id for s in BASE_MODELS.values()]
        self.assertEqual(len(ids), len(set(ids)))

    def test_main_pair_is_the_two_independent_tsfms(self):
        self.assertEqual(set(MAIN_BASES), {"chronos2", "timesfm25"})
        self.assertEqual(set(SENSITIVITY_BASES), {"chronos2_synth"})

    def test_every_family_has_an_adapter(self):
        for spec in BASE_MODELS.values():
            self.assertIn(spec.family, {"chronos", "timesfm"})

    def test_unknown_key_fails_loudly(self):
        with self.assertRaisesRegex(KeyError, "unknown base"):
            get_base("chronos-bolt-base")


class TestDatasetScope(unittest.TestCase):
    def test_main_bases_run_on_both_datasets(self):
        for base in MAIN_BASES:
            for dataset in ("walmart", "gefcom2017f"):
                self.assertEqual(check_base_dataset(base, dataset).key, base)

    def test_synth_arm_is_allowed_on_walmart(self):
        self.assertEqual(check_base_dataset("chronos2_synth", "walmart").role,
                         "sensitivity")

    def test_synth_arm_is_refused_on_gefcom(self):
        with self.assertRaises(SystemExit):
            check_base_dataset("chronos2_synth", "gefcom2017f")

    def test_scope_is_declared_in_the_registry_not_only_in_prose(self):
        self.assertEqual(BASE_MODELS["chronos2_synth"].allowed_datasets, ("walmart",))
        for base in MAIN_BASES:
            self.assertIsNone(BASE_MODELS[base].allowed_datasets)


if __name__ == "__main__":
    unittest.main()
