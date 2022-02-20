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

ID_TO_PRETTY_NAME = {
    1: "Periodic",
    2: "Incremental periodic",
    5: "Basic QA",
    3: "Symbol counting",
    4: "Pattern counting",
    6: "Harder QA",
    7: "QA with world definition",
    8: "QA with world definition and counting",
    9: "Adjective QA",
    10: "Adjective QA and counting",
}

NAME_TO_ID = {
    "periodic": 1,
    "inc-per": 2,
    "sym-ct": 3,
    "hard-sym-ct": 4,
    "qa": 5,
    "hard-qa": 6,
    "qa-world-def": 7,
    "qa-world-def-ct": 8,
    "adj-qa": 9,
    "adj-qa-ct": 10,
}
