# Incremental tasks

[![PyPI Version][pypi-image]][pypi-url]
[![Incremental tasks CI](https://github.com/hugcis/incremental_tasks/actions/workflows/build.yml/badge.svg)](https://github.com/hugcis/incremental_tasks/actions/workflows/build.yml)
[![][versions-image]][versions-url]

<!-- Badges: -->

[pypi-image]: https://img.shields.io/pypi/v/incremental_tasks
[pypi-url]: https://pypi.org/project/incremental_tasks/
[versions-image]: https://img.shields.io/pypi/pyversions/incremental_tasks/
[versions-url]: https://pypi.org/project/incremental_tasks/

This is a modular and extendable benchmark of progressively more difficult AI tasks to measure learning speed of ML systems.

This repository contains the code to generate the incremental task dataset used
in [[1]](#ref).
    

## Installation

This package can also be used as a library. Just install it from PyPI (ideally
in a virtual env if you don't want the CLI command to pollute your path).

```bash
pip install incremental_tasks
```
This installs the library as well as an executable `generate_tasks_cli`

## Task generation

The command `generate_tasks_cli` can be used to directly generate sequences from
the command line. They are printed to stdout and can be saved to a file to
quickly create a dataset.


## Interactive task solving

A user can try the tasks by himself by running `generate_tasks_cli`. This will
start an interactive session that will show random examples from the tasks of
the benchmarks, starting from the easiest.

Once a task is solved, it switches to a new harder one.

An example interactive session:

<pre><code>$ generate_tasks_cli  --interactive

======================================================================
0 1 1 0 0 0 1 1 0 1 1 0 0 0 1 1 0 1 1 <b  style="color:blue">{?} {?} {?} {?} {?}</b>
Type you answers (space separated) 0 0 0 1 1
OK!
0 1 1 0 0 0 1 1 0 1 1 0 0 0 1 1 0 1 1 <b style="color:green">0 0 0 1 1</b>

======================================================================
1 0 0 0 1 0 0 0 <b  style="color:blue">{?} {?} {?} {?} {?}</b>
Type you answers (space separated) 0 1 1 1 0
Wrong! right answer was:
1 0 0 0 1 0 0 0 <b style="color:red">1 0 0 0 1</b>
</code></pre>

In [[1]](#ref) the human evaluation score were computed using this interactive
game with the extra flag `--human-eval` which maps every token to a random one
so the player doesn't have any prior knowledge about the text and needs to do
pattern matching like a neural network would.

## Library

You can use the library in your own code to generate the data on the fly: 

``` python
from incremental_tasks import ElementaryLanguageWithWorldDef

task = ElementaryLanguageWithWorldDef()
```
To generate a single sentence from the task use `generate_single`:
``` python
print(task.generate_single())
# This will print (['I', 'DO', 'NOT', 'SMELL', 'PETER', '.', 'DO', 'I', 'SMELL', 'PETER', '?', 'NO'], [11])
```


To generate `n` unique sequences (will be less than `n` if there aren't enough
available unique sequences): 

``` python
task.generate_tasks(max_n_seq=n)
```

A task can also create a generator that will yield an endless stream of
sequences (not necessarily unique):
``` python
task.generate_tasks_generator(max_n_seq=None)
```

### References

- <a name="ref"></a>[1] Cisneros, H., Mikolov, T., & Sivic, J. (2022).
Benchmarking Learning Efficiency in Deep Reservoir Computing. 1st Conference on
Lifelong Learning Agents, Montreal, Canada.
 
