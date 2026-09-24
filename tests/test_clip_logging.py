"""Run with python3 -m unittest discover -s tests -p test_clip_logging.py.

Torch integration test runs when torch is available (CPU only).
"""
import ast
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


logging = load('clip_logging', 'lib/trainer/clip_logging.py')
analysis = load('analyze_clip_grad', 'scripts/analysis/analyze_clip_grad.py')


class ClipLoggingTests(unittest.TestCase):
    def test_global_norm_does_not_determine_clipping(self):
        # Same global norm, different per-parameter clipping decisions.
        a = logging.step_record([.25, .25], .3, 1, 0, math.sqrt(.125))
        b = logging.step_record([math.sqrt(.125), 0], .3, 1, 1, math.sqrt(.125))
        self.assertEqual(a['n_clipped'], 0)
        self.assertEqual(b['n_clipped'], 1)
        summary = analysis.summarize_steps([a, b])
        self.assertEqual(summary['clipped_step_fraction'], .5)
        self.assertEqual(summary['clipped_parameter_fraction'], .25)

    def test_boundary_disabled_empty_nonfinite(self):
        row = logging.step_record([.3], .3, 1, 0, .3)
        self.assertEqual(row['n_above_threshold'], 0)
        self.assertEqual(row['n_clipped'], 1)  # epsilon in actual formula
        self.assertEqual(logging.step_record([2], 0, 1, 0, 2)['coef_min'], 1)
        self.assertEqual(logging.step_record([], .3, 1, 0, 0)['n_clipped'], 0)
        bad = logging.step_record([float('nan')], .3, 1, 0, float('nan'))
        json.dumps(bad, allow_nan=False)
        self.assertEqual(analysis.summarize_steps([bad])['nonfinite_steps'], 1)
        self.assertIsNone(analysis.summarize_steps([bad])['clipped_step_fraction'])

    def test_resume_dates_rank_exclusion_and_telemetry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / 'config.json').write_text('{"clip_grad": 0.3, "num_epoch": 480}')
            for name, date, norm in [('235900', '20260901-235900', 2), ('010000', '20260902-010000', 3), ('020000_rank1', '20260902-020000', 99)]:
                (path / f'log_{name}.txt').write_text(f'[{date}] [INFO] Epoch: 1, train_loss: 1.0, lr: 1e-3, grad_norm: {norm}\n')
            self.assertIsNone(analysis.analyze(path)['clipping'])
            for attempt, norms in [(1, [.1]), (2, [.6])]:
                row = logging.step_record(norms, .3, 1, 0, norms[0])
                (path / f'clip_steps_ep0001_{attempt}.jsonl').write_text(json.dumps(row) + '\n')
            result = analysis.analyze(path)
            self.assertEqual(result['epochs'][0]['grad_norm'], 3)
            self.assertEqual(result['duplicate_epoch_rows'], 1)
            self.assertEqual(result['clipping']['steps'], 1)
            self.assertEqual(result['clipping']['clipped_step_fraction'], 1)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'torch unavailable on host')
    def test_train_epoch_telemetry_preserves_update(self):
        import numpy as np
        import torch
        tree = ast.parse((ROOT / 'lib/trainer/loop.py').read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train_epoch')
        namespace = dict(torch=torch, np=np, RunContext=object,
                         tqdm=lambda loader, **kw: loader,
                         write_clip_epoch=logging.write_epoch)
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<train_epoch>', 'exec'), namespace)
        with tempfile.TemporaryDirectory() as tmp:
            states = []
            for main in (False, True):
                namespace['distributed'] = types.SimpleNamespace(
                    is_main_process=lambda: main, unwrap=lambda model: model,
                    all_reduce_mean=lambda value: value)
                model = torch.nn.Linear(2, 1)
                with torch.no_grad():
                    model.weight.fill_(1)
                    model.bias.fill_(1)
                ctx = types.SimpleNamespace(args=types.SimpleNamespace(clip_grad=.3), dir_name=tmp,
                    ssl_class=types.SimpleNamespace(calc_loss=lambda model, data, criterion: model(data).square().mean()))
                namespace['train_epoch'](ctx, model, [torch.ones(2, 2)] * 2, None,
                                         torch.optim.SGD(model.parameters(), lr=.1), 0, 2)
                states.append([p.detach().clone() for p in model.parameters()])
            for a, b in zip(*states):
                self.assertTrue(torch.equal(a, b))
            files = list(Path(tmp).glob('clip_steps_*.jsonl'))
            self.assertEqual(len(files), 1)
            rows = [json.loads(line) for line in files[0].read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(r['n_clipped'] == 2 for r in rows))


if __name__ == '__main__':
    unittest.main()
