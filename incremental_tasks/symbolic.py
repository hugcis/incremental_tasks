"""This module implements the symbolic tasks."""
import collections
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np

from .base import SingleTM, TokenTask


class SymbolCounting(TokenTask):
    """The base symbol counting task."""

    def __init__(
        self,
        lengths: Optional[Union[int, List[int]]] = None,
        dictionary: Optional[List[str]] = None,
        query_symbol: str = "x",
        eol_symbol: str = ".",
    ):
        if lengths is None:
            lengths = [10]
        if dictionary is None:
            dictionary = ["A", "B", "C"]
        super().__init__(
            "sym-ct",
            lengths,
            dictionary + [query_symbol, eol_symbol] + [str(i) for i in range(10)],
        )
        assert query_symbol not in dictionary
        assert np.all([len(i) == 1 for i in dictionary])
        self.query_symbol = query_symbol
        self.eol_symbol = eol_symbol
        self.base_dic = dictionary

    def generate_single(self, **kwargs) -> SingleTM:
        del kwargs
        base_len = np.random.choice(self.lengths)
        current_task_mask = []
        n_queries = np.random.randint(1, len(self.base_dic) + 1)
        left = np.random.choice(self.base_dic, size=base_len, replace=True).tolist()
        counter: collections.Counter = collections.Counter(left)

        symbol_choice = np.random.choice(self.base_dic, size=n_queries, replace=False)
        for symbol in symbol_choice:
            left = left + [self.query_symbol, symbol] + list(str(counter[symbol]))
            for i in reversed(range(len(str(counter[symbol])))):
                current_task_mask.append(len(left) - 1 - i)
        left.append(self.eol_symbol)
        return left, current_task_mask


@dataclass
class HardSymCountingSymbols:
    """A simple class holding all the extra symbols needed for the pattern
    counting task."""

    separator_symbol: str = "y"
    query_symbol: str = "x"
    eol_symbol: str = "."


class HardSymbolCounting(TokenTask):
    """The 'pattern' counting task. Instead of counting symbols, exact matches
    of groups of symbols must be found

    """

    def __init__(
        self,
        lengths: Optional[Union[int, List[int]]] = None,
        dictionary: Optional[List[str]] = None,
        symbols: HardSymCountingSymbols = HardSymCountingSymbols(),
    ):
        if lengths is None:
            lengths = [45]
        if dictionary is None:
            dictionary = ["A", "B", "C", "D", "E"]

        super().__init__(
            "hard-sym-ct",
            lengths,
            dictionary
            + [symbols.query_symbol, symbols.separator_symbol, symbols.eol_symbol]
            + [str(i) for i in range(10)],
        )
        assert symbols.query_symbol not in dictionary
        assert np.all([len(i) == 1 for i in dictionary])
        self.query_symbol = symbols.query_symbol
        self.separator_symbol = symbols.separator_symbol
        self.base_dic = dictionary + [symbols.separator_symbol]
        self.eol_symbol = symbols.eol_symbol

    def generate_single(self, **kwargs) -> SingleTM:
        del kwargs
        base_len = np.random.choice(self.lengths)
        current_task_mask = []
        left: list[str] = []
        while not left:
            left = np.random.choice(
                self.base_dic + [self.separator_symbol] * 2 * len(self.base_dic),
                size=int(2.5 * base_len),
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
        counter = collections.Counter("".join(left).split(self.separator_symbol))
        n_queries = np.random.randint(1, len(counter.keys()) + 1)

        pattern_choice = np.random.choice(
            list(counter.keys()), size=n_queries, replace=False
        )
        left = left + [self.query_symbol]
        for pattern in pattern_choice:

            left = (
                left
                + list(pattern)
                + [self.separator_symbol]
                + list(str(counter[pattern]))
            )
            for i in range(len(str(counter[pattern]))):
                current_task_mask.append(len(left) - 1 - i)
            if np.random.random() > 0.2:
                negative = list(3 * pattern[:])
                np.random.shuffle(negative)
                negative = negative[: len(pattern) + np.random.randint(-2, 3)]
                if negative and "".join(negative) not in pattern:
                    left = left + negative + [self.separator_symbol, "0"]
                    current_task_mask.append(len(left) - 1)
        left.append(self.eol_symbol)
        return left, current_task_mask
