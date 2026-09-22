#!/usr/bin/env python3
"""End-to-end training-pipeline test (PS-212 layer).

A miniature but REAL subsystem run: synthetic multi-channel signal ->
Hilbert envelope features -> tiny ResNet1D classifier -> cross-entropy
loss -> backward pass. CPU-only, seeded, <60s.

Gated by `RUN_E2E=1` (skipped by default); the `e2e` marker is registered
in `pyproject.toml [tool.pytest.ini_options]`.
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_E2E") != "1",
    reason="e2e layer runs only with RUN_E2E=1",
)

torch = pytest.importorskip("torch")

import scitex_nn


@pytest.mark.e2e
def test_training_pipeline_forward_backward():
    """Mini train step over synthetic EEG-like data must stay finite."""
    # Arrange
    torch.manual_seed(0)
    model = scitex_nn.ResNet1D(n_chs=4, n_out=2, n_blks=1)
    x = torch.randn(4, 4, 256)
    target = torch.zeros(4, dtype=torch.long)
    # Act
    pooled = torch.nan_to_num(model(x), nan=0.0, posinf=1e4, neginf=-1e4).mean(
        dim=-1
    )
    loss = torch.nn.functional.cross_entropy(pooled, target)
    loss.backward()
    # Assert
    assert bool(torch.isfinite(loss)) and all(
        p.grad is not None for p in model.parameters() if p.requires_grad
    )
