"""Official adapter calls must receive the registry's immutable HF revision."""

# Resolve the approved project layout while retaining sealed source-path keys.
from src.support.paths.project import project_file

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_file(ROOT)))

from src.base_forecasts.registry import BASE_MODELS
from src.base_forecasts.chronos2 import load_pipeline
from src.base_forecasts.timesfm25 import load_model


class TestRevisionPassthrough(unittest.TestCase):
    def test_chronos_loader_receives_exact_revision(self):
        calls = {}

        class FakePipeline:
            @classmethod
            def from_pretrained(cls, model_id, **kwargs):
                calls.update(model_id=model_id, **kwargs)
                return object()

        with patch.dict(sys.modules, {"chronos": SimpleNamespace(
                BaseChronosPipeline=FakePipeline)}):
            spec = BASE_MODELS["chronos2"]
            load_pipeline(spec.model_id, "cpu", revision=spec.revision)
        self.assertEqual(calls["revision"], spec.revision)

    def test_timesfm_loader_receives_exact_revision(self):
        calls = {}

        class Inner:
            def __init__(self):
                import torch
                self.device = torch.device("cpu")
                self.device_count = 1

        class FakeModel:
            def __init__(self):
                self.model = Inner()

            def compile(self, config):
                self.config = config

        class FakeWrapper:
            @classmethod
            def from_pretrained(cls, model_id, **kwargs):
                calls.update(model_id=model_id, **kwargs)
                return FakeModel()

        class FakeConfig:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        with patch.dict(sys.modules, {"timesfm": SimpleNamespace(
                TimesFM_2p5_200M_torch=FakeWrapper,
                ForecastConfig=FakeConfig)}):
            spec = BASE_MODELS["timesfm25"]
            load_model(spec.model_id, device="cpu", context_len=64,
                       predict_len=14, batch_size=2, revision=spec.revision)
        self.assertEqual(calls["revision"], spec.revision)


if __name__ == "__main__":
    unittest.main()
