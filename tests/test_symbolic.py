"""Test for the symbolic tasks."""
import numpy as np

from incremental_tasks.symbolic import HardSymbolCounting, SymbolCounting

sequences_symbol_counting = [
    "BBCACAxB2.",
    "CAxA1xC1xB0.",
    "ABAABCACABxA5xC2.",
    "ABBBxC0xB3xA1.",
    "CACAAxC2xB0xA3.",
]

sequences_pattern_counting = [
    "BAyCyDyCyEyDyByDyDyE",
    "DyDyCyAEyCByByByCxAE",
    "AyEByEyDyByDyBxBy2.",
    "DyCyDyDyEyDyBAyDyByC",
    "AyAyEyEyEyDyAyByEDyE",
]


def test_symbol_counting():
    np.random.seed(0)
    task = SymbolCounting()

    for idx in range(5):
        seq = task.generate_single()
        assert "".join(seq[0])[:20] == sequences_symbol_counting[idx]


def test_pattern_counting():
    np.random.seed(0)
    task = HardSymbolCounting()

    for idx in range(5):
        seq = task.generate_single()
        assert "".join(seq[0])[:20] == sequences_pattern_counting[idx]
