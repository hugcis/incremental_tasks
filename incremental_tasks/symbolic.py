import collections
from typing import List, Union

import numpy as np

from .base import SingleTM, TokenTask


class SymbolCounting(TokenTask):
    def __init__(
        self,
        lengths: Union[int, List[int]] = [10],
        dictionary: List[str] = ["A", "B", "C"],
        query_symbol: str = "x",
    ):
        super().__init__(
            "sym-ct", lengths, dictionary + [query_symbol] + [str(i) for i in range(10)]
        )
        assert query_symbol not in dictionary
        assert np.all([len(i) == 1 for i in dictionary])
        self.query_symbol = query_symbol
        self.base_dic = dictionary

    def generate_single(self, **kwargs) -> SingleTM:
        t = np.random.choice(self.lengths)
        current_task_mask = []
        n_queries = np.random.randint(1, len(self.base_dic) + 1)
        left = np.random.choice(self.base_dic, size=t, replace=True).tolist()
        ct: collections.Counter = collections.Counter(left)

        tk = np.random.choice(self.base_dic, size=n_queries, replace=False)
        for tc in tk:
            left = left + [self.query_symbol, tc] + [i for i in str(ct[tc])]
            for i in range(len(str(ct[tc]))):
                current_task_mask.append(len(left) - 1 - i)
        return left, current_task_mask


class HardSymbolCounting(TokenTask):
    def __init__(
        self,
        lengths: Union[int, List[int]] = [45],
        dictionary: List[str] = ["A", "B", "C", "D", "E"],
        separator_symbol: str = "y",
        query_symbol: str = "x",
    ):
        super().__init__(
            "hard-sym-ct",
            lengths,
            dictionary + [query_symbol, separator_symbol] + [str(i) for i in range(10)],
        )
        assert query_symbol not in dictionary
        assert np.all([len(i) == 1 for i in dictionary])
        self.query_symbol = query_symbol
        self.separator_symbol = separator_symbol
        self.base_dic = dictionary + [separator_symbol]

    def generate_single(self, **kwargs) -> SingleTM:
        t = np.random.choice(self.lengths)
        current_task_mask = []
        left: list[str] = []
        while not left:
            left = np.random.choice(
                self.base_dic + [self.separator_symbol] * 2 * len(self.base_dic),
                size=int(2.5 * t),
                replace=True,
            ).tolist()
            while left and left[0] == self.separator_symbol:
                left.pop(0)
            while left and left[-1] == self.separator_symbol:
                left.pop(0)
            # No duplicates separator symbols
            left = [
                v
                for i, v in enumerate(left)
                if (
                    i == 0
                    or (v == self.separator_symbol and v != left[i - 1])
                    or v != self.separator_symbol
                )
            ]

        # Add the query part
        ct = collections.Counter("".join(left).split(self.separator_symbol))
        n_queries = np.random.randint(1, len(ct.keys()) + 1)

        tk = np.random.choice(list(ct.keys()), size=n_queries, replace=False)
        left = left + [self.query_symbol]
        for tc in tk:

            left = left + list(tc) + [self.separator_symbol] + [i for i in str(ct[tc])]
            for i in range(len(str(ct[tc]))):
                current_task_mask.append(len(left) - 1 - i)
            if np.random.random() > 0.2:
                negative = list(3 * tc[:])
                np.random.shuffle(negative)
                negative = negative[: len(tc) + np.random.randint(-2, 3)]
                if negative and not "".join(negative) in ct:
                    left = left + negative + [self.separator_symbol, "0"]
                    current_task_mask.append(len(left) - 1)
        return left, current_task_mask
