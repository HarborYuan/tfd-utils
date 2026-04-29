from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("tfd-utils")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

def hello() -> str:
    return "Hello from tfd-utils!"

from .random_access import TFRecordRandomAccess
from .tar_random_access import TarRandomAccess

__all__ = ['TFRecordRandomAccess', 'TarRandomAccess', 'hello', '__version__']
