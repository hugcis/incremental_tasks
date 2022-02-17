"""The task library entrypoint."""
from .language import (
    AdjectiveLanguage,
    ElementaryLanguage,
    ElementaryLanguageWithWorldDef,
    ElementaryLanguageWithWorldDefCounting,
    HarderElementaryLanguage,
    AdjectiveLanguageCounting,
)
from .periodic import IncreasingPeriod, Periodic, RandomPeriodic
from .symbolic import HardSymbolCounting, SymbolCounting
from .base import (
    BinarizedTask,
    BinaryTask,
    HybridTask,
    Mask,
    Task,
    TaskMask,
    TaskType,
    TokenTask,
)

__all__ = [
    "Task",
    "BinarizedTask",
    "TokenTask",
    "BinaryTask",
    "TaskMask",
    "Mask",
    "SymbolCounting",
    "HardSymbolCounting",
    "Periodic",
    "IncreasingPeriod",
    "RandomPeriodic",
    "ElementaryLanguage",
    "ElementaryLanguageWithWorldDef",
    "ElementaryLanguageWithWorldDefCounting",
    "HarderElementaryLanguage",
    "AdjectiveLanguage",
    "HybridTask",
    "TaskType",
    "AdjectiveLanguageCounting",
]

ID_TO_TASK = {
    1: Periodic,
    2: IncreasingPeriod,
    5: ElementaryLanguage,
    3: SymbolCounting,
    4: HardSymbolCounting,
    6: HarderElementaryLanguage,
    7: ElementaryLanguageWithWorldDef,
    8: ElementaryLanguageWithWorldDefCounting,
    9: AdjectiveLanguage,
    10: AdjectiveLanguageCounting,
}

NAME_TO_TASK = {
    "periodic": Periodic,
    "inc-per": IncreasingPeriod,
    "sym-ct": SymbolCounting,
    "hard-sym-ct": HardSymbolCounting,
    "qa": ElementaryLanguage,
    "hard-qa": HarderElementaryLanguage,
    "qa-world-def": ElementaryLanguageWithWorldDef,
    "qa-world-def-ct": ElementaryLanguageWithWorldDefCounting,
    "adj-qa": AdjectiveLanguage,
    "adj-qa-ct": AdjectiveLanguageCounting,
}
