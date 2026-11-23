"""Guards on frozen-forecast reuse (numpy only).

Frozen base forecasts are expensive to regenerate and a stale .npz looks
exactly like a fresh one.  Every check below corresponds to a way the pipeline
could silently produce a plausible but meaningless reconciliation table:
forecasts from the retired five-segment split, from a different dataset, from a
different stride, with NaNs baked in, or — the one that would break RQ-D
outright — from a DIFFERENT base than the filename claims.
"""

# Resolve the approved project layout while retaining sealed source-path keys.
from src.support.paths.project import project_file

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_file(ROOT)))

from src.data.datasets import SPLIT_PROTOCOL, SplitOrigins, targets_for_origins
from src.support.storage.forecasts import check_forecast_block, compute_base_fingerprint, load_base_forecasts, save_base_forecasts

T, N_NODES, H, L = 400, 5, 4, 20


def _fake_data():
    """A minimal stand-in for HTSData with a deterministic 80/10/10 geometry."""
    rng = np.random.default_rng(0)
    Y = rng.uniform(1.0, 9.0, size=(T, N_NODES))
    origins = SplitOrigins(
        train=np.arange(L - 1, 320 - H, dtype=int),
        val=np.arange(319, 360 - H, dtype=int),
        test=np.arange(359, T - H, dtype=int),
    )
    return SimpleNamespace(
        name="toy", Y=Y, context_len=L, predict_len=H,
        hierarchy=SimpleNamespace(n=N_NODES),
        cfg={"nonneg": True},
        origins=lambda: origins,
    ), origins


def _payload(data, origins, noise=0.0):
    payload = {}
    rng = np.random.default_rng(1)
    for seg in ("train", "val", "test"):
        arr = getattr(origins, seg)
        ytrue = targets_for_origins(data.Y, arr, H)
        payload[f"origins_{seg}"] = arr
        payload[f"ytrue_{seg}"] = ytrue
        payload[f"yhat_{seg}"] = ytrue + noise * rng.normal(size=ytrue.shape)
    return payload


def _meta(**over):
    meta = {"dataset": "toy", "base": "chronos2", "model_id": "amazon/chronos-2",
            "role": "main", "split_protocol": SPLIT_PROTOCOL,
            "gradient_updates": 0, "context_len": L, "predict_len": H,
            "point_statistic": "q50"}
    meta.update(over)
    return meta


class TestLoadBaseForecasts(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.path = self.tmp / "toy_chronos2.npz"
        self.data, self.origins = _fake_data()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _save(self, payload=None, meta=None):
        payload = _payload(self.data, self.origins) if payload is None else payload
        meta = _meta() if meta is None else meta
        # step1 records the fingerprint; mirror that so the load-time
        # sidecar-vs-arrays consistency check has something to verify.
        if "fingerprint" not in meta:
            meta["fingerprint"] = compute_base_fingerprint(payload)
        save_base_forecasts(self.path, payload, meta)

    def test_matching_file_loads(self):
        self._save()
        fz = load_base_forecasts(self.path, self.data, "chronos2")
        self.assertEqual(fz["yhat_test"].shape,
                         (self.origins.test.size, N_NODES, H))

    def test_old_split_protocol_is_rejected(self):
        self._save(meta=_meta(split_protocol="five_segment_base_head_split"))
        with self.assertRaisesRegex(ValueError, "split_protocol"):
            load_base_forecasts(self.path, self.data, "chronos2")

    def test_missing_split_protocol_is_rejected(self):
        meta = _meta()
        del meta["split_protocol"]
        self._save(meta=meta)
        with self.assertRaisesRegex(ValueError, "split_protocol"):
            load_base_forecasts(self.path, self.data, "chronos2")

    def test_missing_metadata_sidecar_is_rejected(self):
        self._save()
        self.path.with_suffix(".meta.json").unlink()
        with self.assertRaises(FileNotFoundError):
            load_base_forecasts(self.path, self.data, "chronos2")

    def test_wrong_dataset_is_rejected(self):
        self._save(meta=_meta(dataset="gefcom2017f"))
        with self.assertRaisesRegex(ValueError, "dataset"):
            load_base_forecasts(self.path, self.data, "chronos2")

    def test_wrong_context_len_is_rejected(self):
        self._save(meta=_meta(context_len=L + 1))
        with self.assertRaisesRegex(ValueError, "context_len"):
            load_base_forecasts(self.path, self.data, "chronos2")

    def test_a_trained_base_is_rejected(self):
        self._save(meta=_meta(gradient_updates=1200))
        with self.assertRaisesRegex(ValueError, "gradient"):
            load_base_forecasts(self.path, self.data, "chronos2")

    def test_changed_stride_is_rejected(self):
        thinned = SplitOrigins(train=self.origins.train[::4], val=self.origins.val,
                               test=self.origins.test)
        self._save(payload=_payload(self.data, thinned))
        with self.assertRaisesRegex(ValueError, "origins_train"):
            load_base_forecasts(self.path, self.data, "chronos2")

    def test_float_origin_storage_is_rejected_before_lossy_cast(self):
        payload = _payload(self.data, self.origins)
        payload["origins_train"] = payload["origins_train"].astype(np.float64)
        self._save(payload=payload)
        with self.assertRaisesRegex(ValueError, "integer dtype"):
            load_base_forecasts(self.path, self.data, "chronos2")

    def test_nonfinite_origin_storage_is_rejected_before_cast(self):
        payload = _payload(self.data, self.origins)
        payload["origins_train"] = payload["origins_train"].astype(np.float64)
        payload["origins_train"][0] = np.nan
        self._save(payload=payload, meta=_meta(fingerprint="0" * 32))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            load_base_forecasts(self.path, self.data, "chronos2")

    def test_ytrue_from_different_data_is_rejected(self):
        payload = _payload(self.data, self.origins)
        payload["ytrue_val"] = payload["ytrue_val"] + 1.0
        self._save(payload=payload)
        with self.assertRaisesRegex(ValueError, "ytrue_val"):
            load_base_forecasts(self.path, self.data, "chronos2")

    def test_non_finite_forecasts_are_rejected(self):
        payload = _payload(self.data, self.origins)
        payload["yhat_test"][3, 2, 1] = np.nan
        self._save(payload=payload)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            load_base_forecasts(self.path, self.data, "chronos2")

    def test_missing_fingerprint_is_rejected(self):
        self._save(meta=_meta(fingerprint=None))
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            load_base_forecasts(self.path, self.data, "chronos2")

    def test_sidecar_that_drifted_from_the_arrays_is_rejected(self):
        # Guards the cheap `stored_base_fingerprint` path used by later stages:
        # if the .npz were edited without the sidecar, they would trust a hash
        # that no longer describes the data.
        self._save(meta=_meta(fingerprint="0" * 32))
        with self.assertRaisesRegex(ValueError, "diverged"):
            load_base_forecasts(self.path, self.data, "chronos2")

    def test_metadata_sidecar_records_the_protocol(self):
        self._save()
        meta = json.loads(project_file(self.path.with_suffix(".meta.json")).read_text(encoding='utf-8'))
        self.assertEqual(meta["split_protocol"], SPLIT_PROTOCOL)
        self.assertEqual(meta["gradient_updates"], 0)


class TestBaseIdentity(unittest.TestCase):
    """A file must hold the forecasts of the base the caller asked for.

    Without these, `{dataset}_timesfm25.npz` could contain Chronos output and
    the cross-base ranking would compare a model against itself.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.data, self.origins = _fake_data()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _save(self, filename, **meta_over):
        path = self.tmp / filename
        payload = _payload(self.data, self.origins)
        meta = _meta(**meta_over)
        meta.setdefault("fingerprint", compute_base_fingerprint(payload))
        save_base_forecasts(path, payload, meta)
        return path

    def test_chronos_forecasts_filed_under_timesfm_are_rejected(self):
        path = self._save("toy_timesfm25.npz")   # metadata still says chronos2
        with self.assertRaisesRegex(ValueError, "base"):
            load_base_forecasts(path, self.data, "timesfm25")

    def test_requested_base_must_match_metadata(self):
        path = self._save("toy_chronos2.npz")
        with self.assertRaisesRegex(ValueError, "base"):
            load_base_forecasts(path, self.data, "chronos2_synth")

    def test_filename_must_match_dataset_and_base(self):
        path = self._save("wrongname_chronos2.npz")
        with self.assertRaisesRegex(ValueError, "filename"):
            load_base_forecasts(path, self.data, "chronos2")

    def test_reader_namespace_matches_its_metadata(self):
        path = self._save('toy_chronos2__experiment-reader.npz', experiment_id='reader')
        loaded = load_base_forecasts(path, self.data, 'chronos2', segments=('train','val'))
        self.assertIn('yhat_train', loaded)
        self.assertNotIn('yhat_test', loaded)

    def test_reader_namespace_cannot_borrow_another_experiment_identity(self):
        path = self._save('toy_chronos2__experiment-reader.npz', experiment_id='different')
        with self.assertRaisesRegex(ValueError, 'filename'):
            load_base_forecasts(path, self.data, 'chronos2')

    def test_wrong_model_id_for_the_registered_base_is_rejected(self):
        path = self._save("toy_chronos2.npz", model_id="autogluon/chronos-2-synth")
        with self.assertRaisesRegex(ValueError, "model_id"):
            load_base_forecasts(path, self.data, "chronos2")

    def test_wrong_role_is_rejected(self):
        path = self._save("toy_chronos2.npz", role="sensitivity")
        with self.assertRaisesRegex(ValueError, "role"):
            load_base_forecasts(path, self.data, "chronos2")

    def test_non_q50_point_statistic_is_rejected(self):
        path = self._save("toy_chronos2.npz", point_statistic="mean")
        with self.assertRaisesRegex(ValueError, "point_statistic"):
            load_base_forecasts(path, self.data, "chronos2")

    def test_missing_point_statistic_is_rejected(self):
        path = self.tmp / "toy_chronos2.npz"
        meta = _meta()
        del meta["point_statistic"]
        save_base_forecasts(path, _payload(self.data, self.origins), meta)
        with self.assertRaisesRegex(ValueError, "point_statistic"):
            load_base_forecasts(path, self.data, "chronos2")


class TestCheckForecastBlock(unittest.TestCase):
    def test_accepts_a_clean_block(self):
        check_forecast_block(np.ones((2, 3, 4)), (2, 3, 4), nonneg=True, where="t")

    def test_rejects_a_wrong_shape(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            check_forecast_block(np.ones((2, 3, 4)), (2, 3, 5), nonneg=False, where="t")

    def test_rejects_inf_and_names_the_environment(self):
        block = np.ones((2, 2, 2))
        block[0, 0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "non-finite"):
            check_forecast_block(block, (2, 2, 2), nonneg=False, where="t")

    def test_rejects_negative_values_only_when_nonneg_is_required(self):
        block = -np.ones((2, 2, 2))
        check_forecast_block(block, (2, 2, 2), nonneg=False, where="t")
        with self.assertRaisesRegex(ValueError, "negative"):
            check_forecast_block(block, (2, 2, 2), nonneg=True, where="t")


if __name__ == "__main__":
    unittest.main()
