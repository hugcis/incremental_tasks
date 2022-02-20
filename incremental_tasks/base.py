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
    else:
        if mask is not None and len(mask) != len(tasks):
            return tasks, None
        else:
            return tasks, mask


def get_idx(task: List[str], dictionary: List[str]) -> List[int]:
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

    @abstractmethod
    def generate_single(self, **kwargs) -> SingleTM:
        raise NotImplementedError

    def generate_tasks(self, max_n_seq: int = 10, **kwargs) -> Tuple[TaskType, Mask]:
        tasks = []
        masks: List[List[int]] = []
        st: Set[str] = set()
        ct = 0
        while len(st) < max_n_seq and ct < 5 * max_n_seq:
            task_list, mask = self.generate_single(**kwargs)
            if hasattr(self, "separator_symbol"):
                task_str = getattr(self, "separator_symbol").join(task_list)
            else:
                task_str = "".join(task_list)
            if task_str not in st:
                if mask is not None:
                    masks.append(mask)
                tasks.append(task_list)
                st.add(task_str)
            ct += 1
        return choose_minimal_set(tasks, max_n_seq, mask=masks)

    def generate_tasks_generator(
        self, max_n_seq: int = 10, **kwargs
    ) -> Generator[SingleTM, None, None]:
        count = 0
        while count < max_n_seq:
            yield self.generate_single(**kwargs)
            count += 1

    def get_true_output_size(self) -> int:
        output_space: Set[str] = set()
        sample_tasks, sample_masks = self.generate_tasks(max_n_seq=500)
        if sample_masks is not None:
            for i, task in enumerate(sample_tasks):
                output_space.update(task[k] for k in sample_masks[i])
        return len(output_space)

    def get_n_items_per_seq(self) -> float:
        _, sample_masks = self.generate_tasks(max_n_seq=500)
        if sample_masks is not None:
            return np.mean([len(i) for i in sample_masks])
        else:
            raise ValueError(
                "Cannot estimate number of items per sequence without masking"
            )

    def output_dimension(self) -> int:
        return len(self.dictionary)

    def set_lengths(self, lengths: Union[int, List[int]]):
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
    def __init__(
        self, named_tasks: Dict[str, Type[Task]], task_args: Dict[str, List[Any]]
    ):
        self.named_tasks = {}
        # With this, we create a new instance of each subtask every time we
        # create a new HybridTask
        for n in named_tasks:
            self.named_tasks[n] = named_tasks[n](*task_args.get(n, []))
        super().__init__(
            "hyb_{}".format("_".join(t.name for t in self.named_tasks.values()))
        )

        # The dictionary is the union of all subtask dictionaries
        set_dictionary = set()
        for task in self.named_tasks.values():
            set_dictionary.update(task.dictionary)
        self.dictionary = list(set_dictionary)

    def generate_tasks(self, max_n_seq: int = 10, **kwargs) -> TaskMask:
        res: TaskType = []
        msk: Mask = None
        # Each task contributes a fraction of the total sequences
        max_n_per_task = max_n_seq // len(self.named_tasks)
        for n in self.named_tasks:
            task, mask = self.named_tasks[n].generate_tasks(
                max_n_seq=max_n_per_task, **kwargs
            )
            res = res + task
            # TODO Take care of cases where some tasks have masks and others
            # don't
            if mask is not None:
                if msk is None:
                    msk = []
                msk = msk + mask
        return res, msk


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
        dictionary: Sequence[str] = ["A", "B", "C"],
    ):
        super().__init__(name)
        self.dictionary = list(dictionary)
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

    def convert_to_binary(self, task: TaskType, mask: Mask) -> TaskMask:
        task = [[c for g in t for c in self.mapping[g]] for t in task]
        if mask is not None:
            ret_mask = [
                [self.enc_size * c + i for i in range(self.enc_size) for c in m]
                for m in mask
            ]
        else:
            ret_mask = None
        return task, ret_mask

    def generate_tasks(self, max_n_seq: int, **kwargs) -> TaskMask:
        task, mask = self.base_task.generate_tasks(max_n_seq=max_n_seq, **kwargs)
        return self.convert_to_binary(task, mask)


def print_with_sep(tasks, sep="", lim=10):
    """Pretty print some examples from a task's generated sequences."""
    print("\n".join([sep.join([str(k) for k in s]) for s in tasks[:lim]]))
