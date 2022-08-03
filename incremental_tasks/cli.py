#!/usr/bin/env python3
"""The task generator. Can be used as a command line dataset generator or as an
interactive game.

"""
import random
import string
import sys
from argparse import ArgumentParser
from typing import Callable, List, Sequence, Tuple, Union

import numpy as np

from incremental_tasks import ID_TO_TASK, NAME_TO_ID, Task
from incremental_tasks._version import __version__
from incremental_tasks.base import get_idx


def make_parser() -> ArgumentParser:
    """This function creates the argument parser for the CLI tool."""
    parser = ArgumentParser(
        description="This is the incremental tasks generator. "
        "You can use it to generate the tasks and use the benchmark."
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "-i",
        "--task_id",
        metavar="ID",
        type=str,
        default=None,
        help="The ID of the task to generate sentences for.",
    )
    parser.add_argument(
        "-n",
        "--n-examples",
        type=int,
        default=1000,
        help="Maximum number of sentences to generate (actual number generated "
        "might be lower for tasks "
        "with a low number of distinct examples).",
    )
    parser.add_argument(
        "--extra-args",
        type=List[str],
        nargs="*",
        help="Optional additional arguments to be passed to the task instance "
        "(see documentation for details).",
    )
    parser.add_argument(
        "--human-eval",
        action="store_true",
        help="The human evaluation generation mode converts all human readable "
        "symbols into a random symbol to simulate how a predictive model has no "
        "prior notion of language to solve the tasks",
    )
    parser.add_argument(
        "--seed", type=int, help="Provide a random seed for reproducibility"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Try to solve the tasks yourself with this interactive game",
    )
    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="Hide masks from output",
    )

    return parser


def get_task(task_id: Union[str, int]) -> Task:
    """Returns a task instance corresponding to the desired task ID (int) or
    task name (string).

    """
    if isinstance(task_id, int) or task_id.isdigit():
        task_id_int = int(task_id)
    else:
        task_id_int = NAME_TO_ID[task_id]
    return ID_TO_TASK[task_id_int]()


def check_answer(
    answer: List[str], task_list: List[str], mask: List[int]
) -> Tuple[List[str], bool]:
    """Check a user provided answer against the true output."""

    as_task_list = []
    count = 0
    good = True
    for idx, symbol in enumerate(task_list):
        if idx not in mask:
            as_task_list.append(symbol)
        else:
            base = f"\033[1m{symbol}\033[0m\033[0m"
            if count < len(answer) and symbol == answer[count]:
                as_task_list.append("\033[92m" + base)
            else:
                as_task_list.append("\033[91m" + base)
                good = False
            count += 1
    return as_task_list, good


def interactive_session(
    task_id: int, gen_fn: Callable[[], Tuple[Sequence[str], Union[List[int], None]]]
):
    """Runs an interactive task solving session starting from the task with id
    `task_id`.

    The generating function `gen_fn` is used to optionally remap the symbols in
    the sentences.

    """
    if task_id is None:
        current_id = 1
    else:
        current_id = task_id
    while current_id < max(ID_TO_TASK.keys()):
        correct_in_a_row = 0
        n_tries = 0
        wade = {}
        while correct_in_a_row < 5:
            task_gen, mask = gen_fn()
            task_list = list(task_gen)
            assert mask is not None
            mask = sorted(mask)
            if len(mask) > 6:
                stop_idx = mask[5]
                mask = mask[:6]
                task_list = task_list[:stop_idx]

            qs_task_list = [
                s if n not in mask else "\033[94m\033[1m{?}\033[0m\033[0m"
                for n, s in enumerate(task_list)
            ]
            sys.stdout.write(70 * "=" + "\n")
            sys.stdout.write(" ".join(qs_task_list) + "\n")

            answer = input("Type you answers (space separated) ").split(" ")
            as_task_list, good = check_answer(answer, task_list, mask)

            n_tries += 1
            if good:
                sys.stdout.write("OK!\n")
                correct_in_a_row += 1
                if correct_in_a_row / 5 not in wade:
                    wade[correct_in_a_row / 5] = n_tries
            else:
                sys.stdout.write("Wrong! right answer was:\n")
                correct_in_a_row = 0
            sys.stdout.write(" ".join(as_task_list) + "\n\n")
        current_id += 1
        wade_score: float = (1 / sum(i for i in wade)) * sum(
            k / v for k, v in wade.items()
        )
        sys.stdout.write(
            f"It took you {n_tries} sentences! WADE would be {wade_score:.3f} \n"
        )
    print("Congrats you finished the game!")


def main():
    """Main entrypoint for the CLI tool."""
    argparser = make_parser()
    args = argparser.parse_args()
    if hasattr(args, "seed"):
        random.seed(args.seed)
        np.random.seed(args.seed)

    if args.task_id is None:
        task = get_task(random.randint(1, len(ID_TO_TASK)))
    else:
        task = get_task(args.task_id)

    if args.human_eval:
        symbol_map = list(string.ascii_lowercase + string.ascii_uppercase)
        random.shuffle(symbol_map)

        def gen_human_eval():
            t_list, msk = task.generate_single()
            return [symbol_map[x] for x in get_idx(t_list, task.dictionary)], msk

        gen_fn = gen_human_eval
    else:

        def gen_auto():
            t_list, msk = task.generate_single()
            return t_list, msk

        gen_fn = gen_auto

    if args.interactive:
        interactive_session(args.task_id, gen_fn)

    for _ in range(args.n_examples):
        task_list, mask = gen_fn()
        sys.stdout.write(" ".join(task_list) + "\n")
        if mask is not None and not args.no_mask:
            sys.stdout.write(" ".join(map(str, mask)) + "\n")
