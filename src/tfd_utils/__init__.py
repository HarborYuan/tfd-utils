def hello() -> str:
    return "Hello from tfd-utils!"

from .random_access import TFRecordRandomAccess

__all__ = ['TFRecordRandomAccess', 'hello']
