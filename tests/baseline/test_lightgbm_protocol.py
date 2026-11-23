"""The public LightGBM adapter preserves the recorded F03 model parameters."""
import json
from pathlib import Path
from unittest import TestCase

from src.baseline.lightgbm import parameters, protocol


class LightGBMProtocol(TestCase):
    def test_parameters_match_frozen_protocol_for_each_seed_and_leaf_count(self):
        path=Path(__file__).resolve().parents[2]/'configs/experiments/lightgbm_hybn.json'
        frozen=json.loads(path.read_text(encoding='utf-8'))['f03']
        self.assertEqual(protocol(),frozen)
        for seed in (42,43,44):
            for leaves in frozen['num_leaves']:
                expected=dict(frozen['params'],num_leaves=leaves,seed=seed,
                              feature_fraction_seed=seed,bagging_seed=seed,data_random_seed=seed)
                self.assertEqual(parameters(seed,leaves),expected)
