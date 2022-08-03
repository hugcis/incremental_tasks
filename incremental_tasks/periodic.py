"""This module implements the periodic-style tasks."""
from typing import List, Optional, Union

import numpy as np

from .base import BinaryTask, SingleTM


class Periodic(BinaryTask):
    """Generate all binary periodic sequences with lengths."""

    def __init__(self, lengths: Optional[Union[int, List[int]]] = None):
        if lengths is None:
            lengths = [10]
        super().__init__("periodic", lengths)

    def generate_single(self, **kwargs) -> SingleTM:
        seq_len = kwargs.get("seq_len", 100)
        period = np.random.choice(self.lengths)
        sequence = np.random.choice(np.arange(0, 2**period))
        formatted_string = f"{{0:0{period}b}}"
        base = [int(i) for i in formatted_string.format(sequence)]
        task = base * (seq_len // len(base))
        task = (task + base[: seq_len - len(task)])[:]
        task_list = [str(i) for i in task]
        masking_limit = np.random.randint(0, max(period - 1, 1))
        return task_list, list(range(2 * period + masking_limit, len(task_list)))


class IncreasingPeriod(BinaryTask):
    """Generate all binary periodic sequences with increasing periods with
    lengths.
    """

    def __init__(self, lengths: Optional[Union[int, List[int]]] = None):
        if lengths is None:
            lengths = [10]
        super().__init__("inc-per", lengths)

    def generate_single(self, **kwargs) -> SingleTM:
        seq_len = kwargs.get("seq_len", 100)
        base_period = np.random.choice(self.lengths)
        sequence = np.random.choice(np.arange(0, 2**base_period))
        formatted_string = f"{{0:0{base_period}b}}"
        base = [int(i) for i in formatted_string.format(sequence)]
        task = base[:]
        count = 2
        while len(task) < seq_len:
            task = (
                task
                + np.concatenate(
                    [np.array(base)[:, None] for _ in range(count)], axis=1
                )
                .reshape(-1)
                .tolist()
            )
            count += 1

        task = task[:seq_len]
        task_list = [str(i) for i in task]
        masking_limit = np.random.randint(0, max(base_period - 1, 1))
        return task_list, list(range(3 * base_period + masking_limit, len(task_list)))


class RandomPeriodic(BinaryTask):
    """Generate random sequences with N-Grams of length in lenghts."""

    def __init__(self, lengths: Union[int, List[int]]):
        super().__init__("rand-per", lengths)

    def generate_single(self, **kwargs) -> SingleTM:
        seq_len = kwargs.get("seq_len", 100)
        period = np.random.choice(self.lengths)
        random_seq: np.ndarray = np.random.randint(
            2**period, size=(1 + seq_len // period)
        )
        formatted_string = f"{{0:0{period}b}}"
        task = [i for q in random_seq for i in formatted_string.format(q)][:seq_len]

        task_list = [str(i) for i in task]
        return task_list, list(range(1, len(task_list)))
