# Incremental tasks

This repository contains the code to generate the incremental task dataset used
in [[1]](#ref).

## Interactive task solving

A user can try the tasks by himself by running `generate_tasks_cli`. This will
start an interactive session that will show random examples from the tasks of
the benchmarks, starting from the easiest.

Once a task is solved, it switches to a new one.

<pre><code>
======================================================================
0 1 1 0 0 0 1 1 0 1 1 0 0 0 1 1 0 1 1 {?} {?} {?} {?} {?}
Type you answers (space separated) 0 0 0 1 1
OK!
0 1 1 0 0 0 1 1 0 1 1 0 0 0 1 1 0 1 1 0 0 0 1 1

======================================================================
1 0 0 0 1 0 0 0 {?} {?} {?} {?} {?}
Type you answers (space separated) 0 1 1 1 0
Wrong! right answer was:
1 0 0 0 1 0 0 0 1 0 0 0 1
</code></pre>

## Task generation



## Library

This package can be used as a library. Just install it from PyPI

```bash
pip install incremental_tasks
```


### References

- <a name="ref"></a>[1] Cisneros, H., Mikolov, T., & Sivic, J. (2022). Benchmarking Learning
Efficiency in Deep Reservoir Computing. 1st Conference on Lifelong Learning
Agents, Montreal, Canada.
 
