from typing import List, Union

import numpy as np

from .base import BinaryTask, SingleTM


class Periodic(BinaryTask):
    """Generate all binary periodic sequences with lengths."""

    def __init__(self, lengths: Union[int, List[int]] = [10]):
        super().__init__("periodic", lengths)

    def generate_single(self, **kwargs) -> SingleTM:
        seq_len = kwargs.get("seq_len", 100)
        t = np.random.choice(self.lengths)
        s = np.random.choice(np.arange(0, 2**t))
        ft = "{{0:0{}b}}".format(t)
        base = [int(i) for i in ft.format(s)]
        task = base * (seq_len // len(base))
        task = (task + base[: seq_len - len(task)])[:]
        task_list = [str(i) for i in task]
        masking_limit = np.random.randint(0, max(t - 1, 1))
        return task_list, list(range(2 * t + masking_limit, len(task_list)))


class IncreasingPeriod(BinaryTask):
    """Generate all binary periodic sequences with increasing periods with
    lengths.
    """

    def __init__(self, lengths: Union[int, List[int]] = [10]):
        super().__init__("inc-per", lengths)

    def generate_single(self, **kwargs) -> SingleTM:
        seq_len = kwargs.get("seq_len", 100)
        t = np.random.choice(self.lengths)
        s = np.random.choice(np.arange(0, 2**t))
        ft = "{{0:0{}b}}".format(t)
        base = [int(i) for i in ft.format(s)]
        task = base[:]
        ct = 2
        while len(task) < seq_len:
            task = (
                task
                + np.concatenate([np.array(base)[:, None] for _ in range(ct)], axis=1)
                .reshape(-1)
                .tolist()
            )
            ct += 1

        task = task[:seq_len]
        task_list = [str(i) for i in task]
        masking_limit = np.random.randint(0, max(t - 1, 1))
        return task_list, list(range(3 * t + masking_limit, len(task_list)))


class RandomPeriodic(BinaryTask):
    """Generate random sequences with N-Grams of length in lenghts."""

    def __init__(self, lengths: Union[int, List[int]]):
        super().__init__("rand-per", lengths)

    def generate_single(self, **kwargs) -> SingleTM:
        seq_len = kwargs.get("seq_len", 100)
        t = np.random.choice(self.lengths)
        seq = np.random.randint(2**t, size=1 + seq_len // t)
        ft = "{{0:0{}b}}".format(t)
        task = [i for q in seq for i in ft.format(q)][:seq_len]

        task_list = [str(i) for i in task]
        return task_list, list(range(1, len(task_list)))
