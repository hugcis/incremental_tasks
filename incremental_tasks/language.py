"""Implementation of the language based tasks."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import SingleTM, TokenTask


def add_no(verb: str, no_names: Optional[List[str]] = None) -> List[str]:
    """Helper function for the QA tasks. Add NOs affirmation"""
    if no_names is not None and no_names:
        return ["I", "DO", "NOT", verb] + no_names
    return []


def add_yes(verb, yes_names: Optional[List[str]] = None) -> List[str]:
    """Helper function for the QA tasks. Adds YESs affirmations"""
    if yes_names is not None and yes_names:
        return ["I", verb] + yes_names
    return []


def make_sentence(
    verb: str,
    yes_names: Optional[List[str]] = None,
    no_names: Optional[List[str]] = None,
    link_words: Optional[List[str]] = None,
) -> List[str]:
    """Creates a list of sentences of the form YES/NO affirmation linked with
    link words.
    """
    if link_words is None:
        link_words = ["AND", "BUT"]
    base = []
    if np.random.random() < 0.5:
        base += add_no(verb, no_names)
        add = add_yes(verb, yes_names)
        if base and add:
            base += [np.random.choice(link_words)]
            base += add
        elif add:
            base += add
    else:
        base += add_yes(verb, yes_names)
        add = add_no(verb, no_names)
        if base and add:
            base += [np.random.choice(link_words)]
            base += add
        elif add:
            base += add
    return base


@dataclass
class ElementaryLanguageSymbols:
    """A class holding all the necessary symbols for the elementary language
    task."""

    separator_symbol: str = " "
    sentence_term_symbol: str = "."
    query_symbol: str = "?"


class ElementaryLanguage(TokenTask):
    """An elementary language task of the form
    ```
    I [VERB] [NAMES]. DO I [VERB] [NAME] ? YES/NO
    ```
    """

    def __init__(
        self,
        object_names: Optional[List[str]] = None,
        verbs: Optional[List[str]] = None,
        symbols: ElementaryLanguageSymbols = ElementaryLanguageSymbols(),
    ):
        self.object_names = (
            object_names
            if object_names is not None
            else ["PETER", "JOHN", "TOM", "JAMES", "PAUL"]
        )
        self.verbs = verbs if verbs is not None else ["SEE", "HEAR"]
        self.sentence_term_symbol = symbols.sentence_term_symbol
        self.query_symbol = symbols.query_symbol
        self.separator_symbol = symbols.separator_symbol
        dictionary = self.object_names + self.verbs  # + color_adj
        dictionary += [
            "I",
            "DO",
            "NOT",
            "AND",
            "BUT",
            symbols.query_symbol,
            symbols.sentence_term_symbol,
            "YES",
            "NO",
        ]
        super().__init__("qa", 0, dictionary)

    def generate_single(self, **kwargs) -> SingleTM:
        del kwargs
        current_mask = []
        verb = np.random.choice(self.verbs)

        # Choose a subset of object names to work with
        subset_size = np.random.randint(1, len(self.object_names))
        subset = np.random.choice(self.object_names, size=subset_size, replace=False)

        # Decide the yes names and the no names (may be empty)
        yes = np.random.randint(len(subset) + 1)
        yes_names = np.random.choice(subset, size=yes, replace=False).tolist()
        if yes_names:
            yes_names = " AND ".join(yes_names).split(" ")
        no_names = [i for i in subset if i not in yes_names]
        if no_names:
            no_names = " AND ".join(no_names).split(" ")

        # Build the sentence
        task = make_sentence(verb, yes_names, no_names)
        tgt = np.random.choice(subset)
        # Make the answer
        task += (
            [self.sentence_term_symbol]
            + ["DO", "I", verb, tgt, self.query_symbol]
            + ["YES" if tgt in yes_names else "NO"]
        )
        current_mask.append(len(task) - 1)
        return task, current_mask


class HarderElementaryLanguage(ElementaryLanguage):
    """A simple redifinition of the base language tasks in a harder
    configuration with more default object names and verbs.

    """

    def __init__(
        self,
        object_names: Optional[List[str]] = None,
        verbs: Optional[List[str]] = None,
    ):
        super().__init__(
            object_names=object_names
            if object_names is not None
            else [
                "PETER",
                "JOHN",
                "TOM",
                "JAMES",
                "PAUL",
                "MARC",
                "LUKE",
                "SIMON",
                "ANDREW",
                "BRUNO",
                "LISA",
            ],
            verbs=verbs
            if verbs is not None
            else ["SEE", "HEAR", "CALL", "FEEL", "SMELL"],
        )
        self.name = "hard-qa"


class ElementaryLanguageWithWorldDef(ElementaryLanguage):
    """Basic language task with multiple facts stated (world definition)."""

    def __init__(
        self,
        object_names: Optional[List[str]] = None,
        verbs: Optional[List[str]] = None,
    ):
        super().__init__(
            object_names=object_names
            if object_names is not None
            else [
                "PETER",
                "JOHN",
                "TOM",
                "JAMES",
                "PAUL",
                "MARC",
                "LUKE",
                "SIMON",
                "ANDREW",
                "BRUNO",
                "LISA",
                "HENRI",
                "LEO",
            ],
            verbs=verbs
            if verbs is not None
            else [
                "SEE",
                "HEAR",
                "CALL",
                "FEEL",
                "SMELL",
                "UNDERSTAND",
                "TOUCH",
            ],
        )
        self.name = "qa-world-def"

    def make_task_mask_from_name_map(
        self, verbs: np.ndarray, name_map: Dict[str, List[str]]
    ) -> SingleTM:
        """This is an intermediary function that will create a task, mask pair
        from the verbs and name_map of the task."""
        current_mask = []
        task = []
        yes_map: Dict[str, List[str]] = {}
        no_map: Dict[str, List[str]] = {}
        first = True
        for verb in verbs:
            if not first:
                task += [self.sentence_term_symbol]
            else:
                first = False
            # Decide the yes names and the no names (may be empty)
            yes = np.random.randint(len(name_map[verb]) + 1)
            yes_names = np.random.choice(
                name_map[verb], size=yes, replace=False
            ).tolist()
            yes_map[verb] = yes_names
            if yes_names:
                yes_names = " AND ".join(yes_names).split(" ")
            no_names = [i for i in name_map[verb] if i not in yes_names]
            no_map[verb] = no_names
            if no_names:
                no_names = " AND ".join(no_names).split(" ")

            # Build the sentence
            task += make_sentence(verb, yes_names, no_names)

        # Choose which verb/name we will ask about
        question_verb = str(np.random.choice(verbs))
        tgt = np.random.choice(name_map[question_verb])
        # Make the answer
        task += (
            [self.sentence_term_symbol]
            + ["DO", "I", question_verb, tgt, self.query_symbol]
            + ["YES" if tgt in yes_map[question_verb] else "NO"]
        )
        current_mask.append(len(task) - 1)
        return task, current_mask

    def generate_single(self, **kwargs) -> SingleTM:
        del kwargs

        # Choose a subset of object names to work with
        subset_size = np.random.randint(1, len(self.object_names))
        subset = np.random.choice(self.object_names, size=subset_size, replace=False)

        # Choose the number of verbs to use
        n_verbs = np.random.randint(1, min(len(self.verbs), subset_size) + 1)
        verbs = np.random.choice(self.verbs, size=n_verbs, replace=False)

        name_map: Dict[str, List[str]] = {}
        indices = np.random.choice(range(len(subset)), size=n_verbs, replace=False)
        indices = np.sort(indices)
        for i, verb in enumerate(verbs):
            right = len(subset) if i >= len(verbs) - 1 else indices[i + 1]
            name_map[verb] = subset[indices[i] : right].tolist()

        return self.make_task_mask_from_name_map(verbs, name_map)


NUMBERS = [
    "ZERO",
    "ONE",
    "TWO",
    "THREE",
    "FOUR",
    "FIVE",
    "SIX",
    "SEVEN",
    "EIGHT",
    "NINE",
    "TEN",
    "ELEVEN",
    "TWELVE",
]


class ElementaryLanguageWithWorldDefCounting(ElementaryLanguageWithWorldDef):
    """Language task with multiple prompts and some counting questions."""

    def __init__(
        self,
        object_names: Optional[List[str]] = None,
        verbs: Optional[List[str]] = None,
        numbers: Optional[List[str]] = None,
    ):
        super().__init__(
            object_names=object_names
            if object_names is not None
            else [
                "PETER",
                "JOHN",
                "TOM",
                "JAMES",
                "PAUL",
                "MARC",
                "LUKE",
                "SIMON",
                "ANDREW",
                "BRUNO",
                "LISA",
                "HENRI",
                "LEO",
            ],
            verbs=verbs
            if verbs is not None
            else [
                "SEE",
                "HEAR",
                "CALL",
                "FEEL",
                "SMELL",
                "UNDERSTAND",
                "TOUCH",
            ],
        )

        numbers = NUMBERS if numbers is None else numbers
        self.name = "qa-world-def-ct"
        self.number_map = dict(enumerate(numbers))
        self.dictionary += numbers
        self.dictionary += ["HOW", "MANY", "PEOPLE"]

    def make_task_mask_from_name_map(
        self, verbs: np.ndarray, name_map: Dict[str, List[str]]
    ) -> SingleTM:
        """This is an intermediary function that will create a task, mask pair
        from the verbs and name_map of the task."""

        current_mask = []
        task = []
        yes_map: Dict[str, List[str]] = {}
        no_map: Dict[str, List[str]] = {}
        first = True
        for verb in verbs:
            if not first:
                task += [self.sentence_term_symbol]
            else:
                first = False
            # Decide the yes names and the no names (may be empty)
            yes = np.random.randint(len(name_map[verb]) + 1)
            yes_names = np.random.choice(
                name_map[verb], size=yes, replace=False
            ).tolist()
            yes_map[verb] = yes_names
            if yes_names:
                yes_names = " AND ".join(yes_names).split(" ")
            no_names = [i for i in name_map[verb] if i not in yes_names]
            no_map[verb] = no_names
            if no_names:
                no_names = " AND ".join(no_names).split(" ")

            # Build the sentence
            task += make_sentence(verb, yes_names, no_names)

        coin_up = np.random.random() > 0.5
        if coin_up:
            # Choose which verb we will ask about
            question_verb = str(np.random.choice(verbs))
            # Make the answer
            task += (
                [self.sentence_term_symbol]
                + [
                    "HOW",
                    "MANY",
                    "PEOPLE",
                    "DO",
                    "I",
                    question_verb,
                    self.query_symbol,
                ]
                + [self.number_map[len(yes_map[question_verb])]]
            )

        else:
            # Choose which verb/name we will ask about
            question_verb = str(np.random.choice(verbs))
            tgt = np.random.choice(name_map[question_verb])
            # Make the answer
            task += (
                [self.sentence_term_symbol]
                + ["DO", "I", question_verb, tgt, self.query_symbol]
                + ["YES" if tgt in yes_map[question_verb] else "NO"]
            )
        current_mask.append(len(task) - 1)
        return task, current_mask

    def generate_single(self, **kwargs) -> SingleTM:
        del kwargs

        # Choose a subset of object names to work with
        subset_size = np.random.randint(1, len(self.object_names))
        subset = np.random.choice(self.object_names, size=subset_size, replace=False)

        # Choose the number of verbs to use
        n_verbs = np.random.randint(1, min(len(self.verbs), subset_size) + 1)
        verbs = np.random.choice(self.verbs, size=n_verbs, replace=False)

        name_map: Dict[str, List[str]] = {}
        indices = np.random.choice(range(len(subset)), size=n_verbs, replace=False)
        indices = np.sort(indices)
        for i, verb in enumerate(verbs):
            right = len(subset) if i >= len(verbs) - 1 else indices[i + 1]
            name_map[verb] = subset[indices[i] : right].tolist()

        return self.make_task_mask_from_name_map(verbs, name_map)


def make_adj_objs(
    size_adj: List[str], color_adj: List[str], obj_names: List[str]
) -> List[List[str]]:
    """
    Constructs triples of size adjective, color adjective, object name

    Returns:
        A list of lists of objects randomly prefixed.
    """
    output = [[i] for i in np.random.permutation(obj_names)]
    for item in output:
        if np.random.random() > 0.4:
            item.insert(0, np.random.choice(color_adj))
        if np.random.random() > 0.4:
            item.insert(0, np.random.choice(size_adj))
    return output


def make_prefix(name):
    """This selects the correct english prefix depending on the beginning of the
    next name."""
    if name[0] in ["A", "I", "U", "E", "O"]:
        return "AN"
    return "A"


DEFAULT_OBJECT_NAMES = [
    "BANANA",
    "APPLE",
    "PEAR",
    "PEACH",
    "APRICOT",
    "CAR",
    "PLANE",
    "TRAIN",
]


def flatten_and_merge(pre_names: List[List[str]], separator_symbol: str) -> List[str]:
    """This creates a joined and prefixed list of names."""
    flatten_names = [
        make_prefix(prefixed_name[0]) + separator_symbol + " ".join(prefixed_name)
        for prefixed_name in pre_names
    ]
    return " AND ".join(flatten_names).split(" ")


class AdjectiveLanguage(TokenTask):
    """A task with question about adjectives of objects."""

    def __init__(
        self,
        object_names: Optional[List[str]] = None,
        verbs: Optional[List[str]] = None,
        adj: Optional[Tuple[List[str], List[str]]] = None,
        symbols: ElementaryLanguageSymbols = ElementaryLanguageSymbols(),
    ):
        self.object_names = (
            object_names if object_names is not None else DEFAULT_OBJECT_NAMES
        )
        self.color_adj, self.size_adj = (
            adj[0] if adj is not None else ["RED", "GREEN", "BLUE", "YELLOW"],
            adj[1] if adj is not None else ["SMALL", "BIG", "HUGE", "TINY"],
        )
        self.verbs = (
            verbs
            if verbs is not None
            else ["SEE", "HEAR", "CALL", "FEEL", "SMELL", "TOUCH"]
        )
        self.sentence_term_symbol = symbols.sentence_term_symbol
        self.query_symbol = symbols.query_symbol
        self.separator_symbol = symbols.separator_symbol
        dictionary = self.object_names + self.verbs + self.color_adj + self.size_adj
        dictionary += ["I", "DO", "NOT", "AND", "BUT", "WHAT", "A", "AN"]
        dictionary += ["COLOR", "SIZE", "IS", "THE"]
        dictionary += [symbols.query_symbol, symbols.sentence_term_symbol, "YES", "NO"]
        super().__init__("adj-qa", 0, dictionary)

    def make_task_mask_from_name_map(
        self, verbs: List[str], name_map: Dict[str, List[List[str]]]
    ) -> SingleTM:
        """This is an intermediary function that will create a task, mask pair
        from the verbs and name_map of the task."""

        current_mask = []
        task = []
        yes_map: Dict[str, List[List[str]]] = {}
        no_map: Dict[str, List[List[str]]] = {}
        first = True
        for verb in verbs:
            yes_names, no_names = None, None
            if not first:
                task += [self.sentence_term_symbol]
            else:
                first = False
            # Decide the yes names and the no names (may be empty)
            yes = np.random.randint(len(name_map[verb]) + 1)
            pre_yes_names: List[List[str]] = [
                name_map[verb][g]
                for g in np.random.choice(
                    range(len(name_map[verb])), size=yes, replace=False
                )
            ]
            yes_map[verb] = pre_yes_names
            if pre_yes_names:
                yes_names = flatten_and_merge(pre_yes_names, self.separator_symbol)

            pre_no_names = [
                i for i in name_map[verb] if i[0] not in [n[0] for n in pre_yes_names]
            ]
            no_map[verb] = pre_no_names
            if pre_no_names:
                no_names = flatten_and_merge(pre_no_names, self.separator_symbol)

            # Build the sentence
            task += make_sentence(verb, yes_names, no_names)

        # Add the question part
        task += self.construct_question(name_map, yes_map, verbs)

        # Last symbol is the one to predict
        current_mask.append(len(task) - 1)
        return task, current_mask

    def generate_single(self, **kwargs) -> SingleTM:
        del kwargs

        # Choose a subset of object names to work with
        subset_size = np.random.randint(1, len(self.object_names))
        subset: List[str] = np.random.choice(
            self.object_names, size=subset_size, replace=False
        ).tolist()
        adj_subset = make_adj_objs(self.size_adj, self.color_adj, subset)

        # Choose the number of verbs to use
        n_verbs = np.random.randint(1, min(len(self.verbs), subset_size) + 1)
        verbs: List[str] = np.random.choice(
            self.verbs, size=n_verbs, replace=False
        ).tolist()

        name_map: Dict[str, List[List[str]]] = {}
        indices = np.random.choice(range(len(subset)), size=n_verbs, replace=False)
        indices = np.sort(indices)
        for i, verb in enumerate(verbs):
            right = len(subset) if i >= len(verbs) - 1 else indices[i + 1]
            name_map[verb] = adj_subset[indices[i] : right]
        return self.make_task_mask_from_name_map(verbs, name_map)

    def construct_question(
        self,
        name_map: Dict[str, List[List[str]]],
        yes_map: Dict[str, List[List[str]]],
        verbs: List[str],
    ) -> List[str]:
        """An intermediate function to construct a question from a name_map,
        yes_map and verbs."""
        # Choose which verb/name we will ask about
        candidates = [(k, i) for k, c in yes_map.items() for i in c if len(i) > 1]

        # If no possible answer has any adjective, we force the question to be YES/NO
        if candidates:
            question_chooser = np.random.random()
        else:
            question_chooser = 0

        if question_chooser < 1 / (len(self.color_adj) + len(self.size_adj) + 2):
            # YES/NO
            question_verb = str(np.random.choice(verbs))
            tgt: List[str] = name_map[question_verb][
                np.random.randint(len(name_map[question_verb]))
            ]
            # Make the answer
            return (
                [self.sentence_term_symbol]
                + ["DO", "I", question_verb, make_prefix(tgt[0])]
                + tgt
                + [self.query_symbol]
                + ["YES" if tgt in yes_map[question_verb] else "NO"]
            )
        # Question about size or color
        question_verb, tgt = candidates[np.random.randint(len(candidates))]
        # Select the adjective we are asking about
        selected_adj: str = np.random.choice(tgt[:-1])
        if selected_adj in self.color_adj:
            question = ["WHAT", "COLOR", "IS", "THE"]
        elif selected_adj in self.size_adj:
            question = ["WHAT", "SIZE", "IS", "THE"]
        else:
            raise ValueError(f"The adjective {selected_adj} is not in the list.")
        # Make the answer
        return (
            [self.sentence_term_symbol]
            + question
            + [tgt[-1], "I", question_verb, self.query_symbol]
            + [selected_adj]
        )


class AdjectiveLanguageCounting(TokenTask):
    """This is the same as the adjective task with an extra counting component."""

    def __init__(
        self,
        object_names: Optional[List[str]] = None,
        verbs: Optional[List[str]] = None,
        adj: Optional[Tuple[List[str], List[str]]] = None,
        symbols: ElementaryLanguageSymbols = ElementaryLanguageSymbols(),
        n_questions_max: int = 8,
    ):  # pylint: disable=too-many-arguments
        self.object_names = (
            object_names if object_names is not None else DEFAULT_OBJECT_NAMES
        )
        self.color_adj, self.size_adj = (
            adj[0] if adj is not None else ["RED", "GREEN", "BLUE", "YELLOW"],
            adj[1] if adj is not None else ["SMALL", "BIG", "HUGE", "TINY", "NORMAL"],
        )
        self.numbers = dict(enumerate(NUMBERS[: len(self.object_names) + 1]))
        self.verbs = (
            verbs
            if verbs is not None
            else ["SEE", "HEAR", "CALL", "FEEL", "SMELL", "TOUCH"]
        )
        self.symbols = symbols
        self.n_questions_max = n_questions_max
        dictionary = self.object_names + self.verbs + self.color_adj + self.size_adj
        dictionary += ["I", "DO", "NOT", "AND", "BUT", "WHAT", "A", "AN"]
        dictionary += ["COLOR", "SIZE", "IS", "THE"]
        dictionary += ["HOW", "MANY", "THINGS"]
        dictionary += list(self.numbers.values())
        dictionary += [
            self.symbols.query_symbol,
            self.symbols.sentence_term_symbol,
            "YES",
            "NO",
        ]
        super().__init__("adj-qa-ct", 0, dictionary)

    def make_task_map_from_name_map(
        self, verbs: List[str], name_map: Dict[str, List[List[str]]]
    ):
        """This is an intermediary function that will create a task, mask pair
        from the verbs and name_map of the task.

        """
        current_mask = []
        task = []
        yes_map: Dict[str, List[List[str]]] = {}
        no_map: Dict[str, List[List[str]]] = {}
        first = True
        for verb in verbs:
            yes_names, no_names = None, None
            if not first:
                task += [self.symbols.sentence_term_symbol]
            else:
                first = False
            # Decide the yes names and the no names (may be empty)
            yes = np.random.randint(len(name_map[verb]) + 1)
            pre_yes_names: List[List[str]] = [
                name_map[verb][g]
                for g in np.random.choice(
                    range(len(name_map[verb])), size=yes, replace=False
                )
            ]
            yes_map[verb] = pre_yes_names
            if pre_yes_names:
                yes_names = flatten_and_merge(
                    pre_yes_names, self.symbols.separator_symbol
                )
            pre_no_names = [
                i for i in name_map[verb] if i[0] not in [n[0] for n in pre_yes_names]
            ]
            no_map[verb] = pre_no_names
            if pre_no_names:
                no_names = flatten_and_merge(
                    pre_no_names, self.symbols.separator_symbol
                )

            # Build the sentence
            task += make_sentence(verb, yes_names, no_names)

        # Add the question part
        for question in self.question_list(name_map, yes_map, verbs):
            task += question
            # Last symbol is the one to predict
            current_mask.append(len(task) - 1)
        return task, current_mask

    def question_list(
        self,
        name_map: Dict[str, List[List[str]]],
        yes_map: Dict[str, List[List[str]]],
        verbs: List[str],
    ):
        """This intermediate method produces a question list from the name_map,
        yes_map and verb to be added to the final sequence."""

        question_set = set()
        question_list = []
        for _ in range(np.random.randint(1, self.n_questions_max)):
            question = self.construct_question(name_map, yes_map, verbs)
            if self.symbols.separator_symbol.join(question) not in question_set:
                question_set.add(self.symbols.separator_symbol.join(question))
                question_list.append(question)
        return question_list

    def generate_single(self, **kwargs) -> SingleTM:
        del kwargs

        # Choose a subset of object names to work with
        subset_size = np.random.randint(1, len(self.object_names))
        subset: List[str] = np.random.choice(
            self.object_names, size=subset_size, replace=False
        ).tolist()
        adj_subset = make_adj_objs(self.size_adj, self.color_adj, subset)

        # Choose the number of verbs to use
        n_verbs = np.random.randint(1, min(len(self.verbs), subset_size) + 1)
        verbs: List[str] = np.random.choice(
            self.verbs, size=n_verbs, replace=False
        ).tolist()

        name_map: Dict[str, List[List[str]]] = {}
        indices = np.random.choice(range(len(subset)), size=n_verbs, replace=False)
        indices = np.sort(indices)
        for i, verb in enumerate(verbs):
            right = len(subset) if i >= len(verbs) - 1 else indices[i + 1]
            name_map[verb] = adj_subset[indices[i] : right]
        return self.make_task_map_from_name_map(verbs, name_map)

    def construct_question(
        self,
        name_map: Dict[str, List[List[str]]],
        yes_map: Dict[str, List[List[str]]],
        verbs: List[str],
    ) -> List[str]:
        """An intermediate function to construct a question from a name_map,
        yes_map and verbs.

        """
        # Choose which verb/name we will ask about
        candidates = [(k, i) for k, c in yes_map.items() for i in c if len(i) > 1]

        # If no possible answer has any adjective, we force the question to be YES/NO
        if candidates:
            question_chooser = np.random.random()
        else:
            question_chooser = 0
        p_ans = 1 / (len(self.numbers) + len(self.color_adj) + len(self.size_adj) + 2)
        if question_chooser < 2 * p_ans:
            # YES/NO
            question_verb = str(np.random.choice(verbs))
            tgt: List[str] = name_map[question_verb][
                np.random.randint(len(name_map[question_verb]))
            ]
            # Make the answer
            return (
                [self.symbols.sentence_term_symbol]
                + ["DO", "I", question_verb, make_prefix(tgt[0])]
                + tgt
                + [
                    self.symbols.query_symbol,
                    "YES" if tgt in yes_map[question_verb] else "NO",
                ]
            )
        if question_chooser < (2 + len(self.numbers)) * p_ans:
            # HOW MANY ...
            question_verb = str(np.random.choice(verbs))
            number_of_things = self.numbers[len(yes_map[question_verb])]
            # Make the answer
            return [self.symbols.sentence_term_symbol] + [
                "HOW",
                "MANY",
                "THINGS",
                "DO",
                "I",
                question_verb,
                self.symbols.query_symbol,
                number_of_things,
            ]

        # Question about size or color
        question_verb, tgt = candidates[np.random.randint(len(candidates))]
        # Select the adjective we are asking about
        selected_adj: str = np.random.choice(tgt[:-1])
        if selected_adj in self.color_adj:
            question = ["WHAT", "COLOR", "IS", "THE"]
        elif selected_adj in self.size_adj:
            question = ["WHAT", "SIZE", "IS", "THE"]
        else:
            raise ValueError(f"The adjective {selected_adj} is not in the list.")
        # Make the answer
        return (
            [self.symbols.sentence_term_symbol]
            + question
            + [tgt[-1], "I", question_verb, self.symbols.query_symbol]
            + [selected_adj]
        )
