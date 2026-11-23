"""Synthetic end-to-end reader workflow; never evaluates benchmark forecasts."""
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, mock
import tempfile

import numpy as np

from scripts.experiments import run_strategy as cli
from src.data.hierarchy import load_hierarchy


class ReaderWorkflow(TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root/'data.yaml'
        self.config.write_text('dataset: synthetic\n', encoding='utf-8')
        self.source = self.root/'forecasts.npz'
        self.source.write_bytes(b'synthetic cache; arrays supplied below')
        self.source.with_suffix('.meta.json').write_text('{}', encoding='utf-8')
        hierarchy = load_hierarchy(cli.ROOT/'Data/Walmart_4level')
        rng = np.random.default_rng(51)
        self.data = SimpleNamespace(name='walmart', hierarchy=hierarchy,
            eval_steps=[1], train_end=24, cfg={'freq':'D'},
            Y=rng.uniform(1,20,(32,hierarchy.m)) @ hierarchy.S.T)
        self.arrays = {}
        for segment, count in [('train',8),('val',4),('test',3)]:
            self.arrays[f'yhat_{segment}'] = rng.uniform(1,20,(count,hierarchy.n,1))
            self.arrays[f'ytrue_{segment}'] = (rng.uniform(1,20,(count,hierarchy.m)) @ hierarchy.S.T)[:,:,None]
            self.arrays[f'origins_{segment}'] = np.arange(count)
        self.segments = []
        def load(source,data,base,*,segments):
            self.segments.append(segments)
            return {key:value for key,value in self.arrays.items() if key.rsplit('_',1)[-1] in segments}
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch('src.data.datasets.load_config',return_value={}))
        self.stack.enter_context(mock.patch('src.data.datasets.load_dataset',return_value=self.data))
        self.stack.enter_context(mock.patch('src.support.storage.forecasts.load_base_forecasts',side_effect=load))
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.output = self.root/'fit'

    def fit(self):
        args = cli.parser().parse_args(['fit','--config',str(self.config),'--base','chronos2',
            '--forecasts',str(self.source),'--strategy','BU','--lead','1','--output',str(self.output)])
        cli.fit_one(args)

    def evaluate(self, confirmed=True):
        cli.evaluate_one(SimpleNamespace(checkpoint=self.output,device='cpu',confirm_selection_frozen=confirmed))

    def test_fit_excludes_test_and_frozen_evaluation_runs_once(self):
        self.fit()
        self.assertEqual(self.segments,[('train','val')])
        self.evaluate()
        self.assertEqual(self.segments,[('train','val'),('train','test')])
        receipt = json.loads((self.output/'test_evaluation.json').read_text())
        self.assertLess(receipt['coherence_nmae'],0.0001)
        self.assertIn('mase',receipt['level_metrics']['all'])
        with self.assertRaises(FileExistsError): self.evaluate()
        with self.assertRaises(FileExistsError): self.fit()

    def test_unconfirmed_or_changed_inputs_never_open_test_arrays(self):
        self.fit()
        with self.assertRaisesRegex(ValueError,'Freeze'): self.evaluate(False)
        self.source.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'changed'): self.evaluate()
        self.assertEqual(self.segments,[('train','val')])

    def test_interrupted_evaluation_is_not_silently_restarted(self):
        self.fit()
        (self.output/'test_evaluation.started.json').write_text('{}')
        with self.assertRaises(FileExistsError): self.evaluate()
        self.assertEqual(self.segments,[('train','val')])
