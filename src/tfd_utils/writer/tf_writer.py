"""
This file is copied and modifed from 
https://github.com/vahidk/tfrecord/blob/aa3a2e3975cd3be3d47e68be476a2b5c608dde1b/tfrecord/writer.py
It is licensed under MIT License.
"""

"""Writer utils."""

import io
import struct

import crc32c
import numpy as np


class TFRecordWriter:
    """Opens a TFRecord file for writing.

    Params:
    -------
    data_path: str
        Path to the tfrecord file.
    """

    def __init__(self, data_path: str) -> None:
        self.file = io.open(data_path, "wb")

    def __enter__(self):
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and close the file."""
        self.close()
        return False

    def close(self) -> None:
        """Close the tfrecord file."""
        self.file.close()

    def write(
        self,
        record: bytes,
    ) -> None:
        """Write an example into tfrecord file. Either as a Example
        SequenceExample depending on the presence of `sequence_datum`.
        If `sequence_datum` is None (by default), this writes a Example
        to file. Otherwise, it writes a SequenceExample to file, assuming
        `datum` to be the context and `sequence_datum` to be the sequential
        features.

        Params:
        -------
        datum: dict
            Dictionary of tuples of form (value, dtype). dtype can be
            "byte", "float" or "int".
        sequence_datum: dict
            By default, it is set to None. If this value is present, then the
            Dictionary of tuples of the form (value, dtype). dtype can be
            "byte", "float" or "int". value should be the sequential features.
        """
        length = len(record)
        length_bytes = struct.pack("<Q", length)
        self.file.write(length_bytes)
        self.file.write(TFRecordWriter.masked_crc(length_bytes))
        self.file.write(record)
        self.file.write(TFRecordWriter.masked_crc(record))

    @staticmethod
    def masked_crc(data: bytes) -> bytes:
        """CRC checksum."""
        mask = 0xA282EAD8
        crc = crc32c.crc32c(data)
        masked = ((crc >> 15) | (crc << 17)) + mask
        masked = np.uint32(masked & np.iinfo(np.uint32).max)
        masked_bytes = struct.pack("<I", masked)
        return masked_bytes
