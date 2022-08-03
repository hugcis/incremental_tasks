"""Tests for the language tasks."""
from incremental_tasks.language import add_no, add_yes


def test_add_yes():
    verb = "HEAR"
    yes_names = ["TOM", "JAMES"]
    assert add_yes(verb, yes_names) == ["I", "HEAR", "TOM", "JAMES"]
    assert add_yes(verb, None) == []


def test_add_no():
    verb = "HEAR"
    no_names = ["TOM", "JAMES"]
    assert add_no(verb, no_names) == ["I", "DO", "NOT", "HEAR", "TOM", "JAMES"]
    assert add_no(verb, None) == []
