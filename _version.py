"""Store the version of the package within source code."""
import importlib.metadata

try:
    __version__ = importlib.metadata.version("incremental_tasks")
except importlib.metadata.PackageNotFoundError:
    __version__ = "not-installed"
