"""The abstract class that any new task should sub-class. The only mandatory
abstract method is `generate_single` that should generate a SingleTM object: a
2-tuple which has the following elements:

- the first element is the list of tokens of the sentence.
- the second element is an optional list of indexes of tokens that should be
  predicted.

"""
from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

import numpy as np

TaskType = List[List[str]]
Mask = Optional[List[List[int]]]
TaskMask = Tuple[TaskType, Mask]
SingleTM = Tuple[List[str], Optional[List[int]]]


def choose_minimal_set(tasks: TaskType, max_n_seq: int, mask: Mask = None) -> TaskMask:
    """Select `max_n_seq` random task/mask pairs from a list."""
    if len(tasks) > max_n_seq:
        idx = np.random.choice(range(len(tasks)), size=max_n_seq, replace=False)
        if mask is not None and len(mask) == len(tasks):
            return_mask = [mask[i] for i in idx]
        else:
            return_mask = None
        return [tasks[i] for i in idx], return_mask
    if mask is not None and len(mask) != len(tasks):
        return tasks, None
    return tasks, mask


def get_idx(task: List[str], dictionary: List[str]) -> List[int]:
    """Remaps the task symbols to dictionary indexes."""
    return [dictionary.index(s) for s in task]


class Task(ABC):
    """
    Abstract base class for tasks.

    A task should have a dictionary member that contains all the possible
    symbols used in the generated sequences.

    """

    dictionary: List[str]

    def __init__(self, name: str):
        self.name = name
        self.lengths: List[int] = []

    @abstractmethod
    def generate_single(self, **kwargs) -> SingleTM:
        """Generates a single pair of sequence/mask for the parent task."""
        raise NotImplementedError

    def generate_tasks(self, max_n_seq: int = 10, **kwargs) -> Tuple[TaskType, Mask]:
        """This generates at most `max_n_seq` unique sequences (less if no new
        unique sequence is generated)."""
        tasks = []
        masks: List[List[int]] = []
        task_set: Set[str] = set()
        count = 0
        while len(task_set) < max_n_seq and count < 5 * max_n_seq:
            task_list, mask = self.generate_single(**kwargs)
            if hasattr(self, "separator_symbol"):
                task_str = getattr(self, "separator_symbol").join(task_list)
            else:
                task_str = "".join(task_list)
            if task_str not in task_set:
                if mask is not None:
                    masks.append(mask)
                tasks.append(task_list)
                task_set.add(task_str)
            count += 1
        return choose_minimal_set(tasks, max_n_seq, mask=masks)

    def generate_tasks_generator(
        self, max_n_seq: Optional[int] = 10, **kwargs
    ) -> Generator[SingleTM, None, None]:
        """This method returns a generator of tasks that will generate at most
        `max_n_seq` sequences (or infinitly many if `max_n_seq` is `None`).

        """
        count = 0
        while (max_n_seq is not None and count < max_n_seq) or max_n_seq is None:
            yield self.generate_single(**kwargs)
            count += 1

    def get_true_output_size(self) -> int:
        """This method computes the "true" output dictionary size for the given
        task from generated sequences.

        """
        output_space: Set[str] = set()
        sample_tasks, sample_masks = self.generate_tasks(max_n_seq=500)
        if sample_masks is not None:
            for i, task in enumerate(sample_tasks):
                output_space.update(task[k] for k in sample_masks[i])
        return len(output_space)

    def get_n_items_per_seq(self) -> float:
        """This method computes an average number of symbol per sequence (out of
        500 sequences)."""
        _, sample_masks = self.generate_tasks(max_n_seq=500)
        if sample_masks is not None:
            return float(np.mean([len(i) for i in sample_masks]))
        raise ValueError("Cannot estimate number of items per sequence without masking")

    def output_dimension(self) -> int:
        """Returns the output dimension for the task (same as the dictionary
        size)."""
        return len(self.dictionary)

    def set_lengths(self, lengths: Union[int, List[int]]):
        """This is an internal function used to compute the lengths of sequences
        when applicable."""
        if isinstance(lengths, int) or len(lengths) == 1:
            if not isinstance(lengths, int):
                length = lengths[0]
            else:
                length = lengths
            self.lengths = list(range(1, length + 1))
        elif len(lengths) == 2:
            if lengths[1] > lengths[0]:
                self.lengths = list(range(lengths[0], lengths[1]))
            else:
                raise ValueError("Wrong lengths")
        else:
            self.lengths = lengths


class HybridTask(Task):
    """A hybrid task combines multiple existing tasks into one. For this, it
    combines both dictionaries and randomly chooses to generate a sentence from
    one of its original tasks.
    """

    def __init__(
        self, named_tasks: Dict[str, Type[Task]], task_args: Dict[str, List[Any]]
    ):
        self.named_tasks = {}
        # With this, we create a new instance of each subtask every time we
        # create a new HybridTask
        for name in named_tasks:
            self.named_tasks[name] = named_tasks[name](*task_args.get(name, []))
        super().__init__(f"hyb_{'_'.join(t.name for t in self.named_tasks.values())}")

        # The dictionary is the union of all subtask dictionaries
        set_dictionary = set()
        for task in self.named_tasks.values():
            set_dictionary.update(task.dictionary)
        self.dictionary = list(set_dictionary)

    def generate_single(self, **kwargs) -> SingleTM:
        chosen_task = np.random.choice(list(self.named_tasks.keys()))
        return self.named_tasks[chosen_task].generate_single(**kwargs)


class BinaryTask(Task):
    """A task with only two tokens: 0 and 1."""

    def __init__(self, name: str, lengths: Union[int, List[int]]):
        super().__init__(name)
        self.dictionary = ["0", "1"]
        self.set_lengths(lengths)


class TokenTask(Task):
    """A task with tokens."""

    def __init__(
        self,
        name: str,
        lengths: Union[int, List[int]],
        dictionary: Optional[Sequence[str]] = None,
    ):
        super().__init__(name)
        self.dictionary = (
            list(dictionary) if dictionary is not None else ["A", "B", "C"]
        )
        self.set_lengths(lengths)


class BinarizedTask(Task):
    """Binarized version of a token class. This class could be though of as a
    "decorator" that will turn a normal Task into a binarized version
    """

    def __init__(self, base_task: TokenTask):
        super().__init__(f"bin_{base_task.name}")
        self.base_task = base_task
        self.dictionary = ["0", "1"]

        self.enc_size = int(np.ceil(np.log2(len(self.base_task.dictionary))))
        formatter = f"{{:0{self.enc_size}b}}"
        self.mapping = {
            d: formatter.format(n) for n, d in enumerate(self.base_task.dictionary)
        }

    def convert_to_binary(self, task: List[str], mask: Optional[List[int]]) -> SingleTM:
        """This converts a sequence with multiple symbols to a binary one."""
        task = [c for g in task for c in self.mapping[g]]
        if mask is not None:
            ret_mask = [
                self.enc_size * c + i for i in range(self.enc_size) for c in mask
            ]

        else:
            ret_mask = None
        return task, ret_mask

    def generate_single(self, **kwargs) -> SingleTM:
        task, mask = self.base_task.generate_single(**kwargs)
        return self.convert_to_binary(task, mask)


def print_with_sep(tasks, sep="", lim=10):
    """Pretty print some examples from a task's generated sequences."""
    print("\n".join([sep.join([str(k) for k in s]) for s in tasks[:lim]]))
