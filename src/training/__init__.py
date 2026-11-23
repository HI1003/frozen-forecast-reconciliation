"""Train on training origins and select checkpoints on validation origins.

losses: original-scale prediction/coherence objectives.
head_training: shared linear and coherence-budget fitting loop.
model_selection: feasible-first HYBN-epsilon ordering.
mlp_search_training: prediction-only checkpoint selection for MLP candidates.

Dataset-wide search grids and selection live in scripts/experiments/paper_protocol.py.
"""
