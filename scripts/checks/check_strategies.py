"""Fast, CPU-only checks of all nine strategies on synthetic forecast inputs.

Uses the published Walmart hierarchy. It does not load model checkpoints,
train on either benchmark, or reproduce the paper's accuracy results.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def check():
    import numpy as np
    import torch
    from src.data.hierarchy import load_hierarchy
    from src.strategies import HEAD_NAMES, build_strategy

    hierarchy = load_hierarchy(ROOT / 'Data/Walmart_4level')
    rng = np.random.default_rng(42)
    training = rng.uniform(1, 20, size=(16, hierarchy.n))
    inputs = torch.tensor(rng.uniform(-2, 20, size=(5, hierarchy.n)), dtype=torch.float32)
    for name in HEAD_NAMES:
        model = build_strategy(name, hierarchy, training, hidden=(8,), dropout=0).eval()
        outputs = model(inputs)
        assert outputs.shape == inputs.shape and torch.isfinite(outputs).all(), name
        assert (outputs >= 0).all(), name
        if name != 'HYBN':
            torch.testing.assert_close(outputs, outputs[:, hierarchy.bottom_idx] @ model.S.T,
                                       rtol=1e-5, atol=1e-5)
        if name in ('TD', 'TDL', 'TDN'):
            torch.testing.assert_close(outputs[:, hierarchy.top_idx],
                                       inputs[:, hierarchy.top_idx].relu(), rtol=1e-5, atol=1e-5)
    print('All nine strategies passed shape, nonnegativity, coherence and top-total checks.')


if __name__ == '__main__':
    check()
