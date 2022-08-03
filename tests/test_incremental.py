"""Test general package things."""
from incremental_tasks import __version__


def test_version():
    assert __version__ == "0.1.3"
