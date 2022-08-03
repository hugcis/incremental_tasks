"""Tests for the periodic tasks"""
import numpy as np

from incremental_tasks.periodic import IncreasingPeriod, Periodic

sequences_periodic = [
    "00000000000000000000",
    "01110111011101110111",
    "01010101010101010101",
    "10111101111011110111",
    "00101100000101100000",
]

sequences_increasing = [
    "00000000000000000000",
    "01110011111100011111",
    "01010011001100011100",
    "10111110011111111100",
    "00101100000001100111",
]


def test_periodic_basic():
    np.random.seed(0)
    task = Periodic()

    for idx in range(5):
        seq = task.generate_single()
        assert "".join(seq[0])[:20] == sequences_periodic[idx]


def test_periodic_increasing():
    np.random.seed(0)
    task = IncreasingPeriod()

    for idx in range(5):
        seq = task.generate_single()
        assert "".join(seq[0])[:20] == sequences_increasing[idx]
