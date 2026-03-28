# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`tfd_utils` is a lightweight Python library providing a unified random-access interface for TensorFlow TFRecord files and tar archives. It requires no TensorFlow dependency for its core functionality.

## Development Setup

This project uses `uv` for package management (not pip).

```bash
uv venv
uv sync --extra dev     # installs core + dev dependencies (pytest, tensorflow, Pillow, requests)
```

## Commands

```bash
# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/unit/test_random_access.py

# Run a single test
uv run pytest tests/unit/test_random_access.py::TestClassName::test_method_name

# Run integration tests only
uv run pytest tests/integration/

# SA-1B integration test (requires env var)
SA1B_DIR=/path/to/sa1b uv run pytest tests/integration/test_sa1b_tar.py

# Build the package
uv build
```

## Architecture

### Two Reader Classes

Both classes share the same public API:

- **`TFRecordRandomAccess`** (`random_access.py`): reads TFRecord files. Each record is keyed by a feature named `'key'` (configurable). `get_record()` returns a parsed `Example` protobuf. Index cached as `<file>.index` (pickled dict: `{key: {file, offset, length}}`). Uses `FileContextPool` (LRU cache of raw file handles) for O(1) seek-based access.

- **`TarRandomAccess`** (`tar_random_access.py`): reads tar archives. Keys are derived from member file stems (`./sa_000001.jpg` → key `sa_000001`, feature `jpg`). `get_record()` returns `dict[str, bytes]`. Index cached as `<file>.tar_index` (pickled dict: `{key: {ext: {file, member_name, data_offset, size}}}`). Uses `TarFilePool` (LRU cache of open `TarFile` objects) with `extractfile()` for access. Supports both uncompressed and compressed (gzip, bzip2) archives via `tarfile.open('r:*')`.

### Other Modules

- **`writer/tf_writer.py`**: `TFRecordWriter` — context manager for writing TFRecord files with CRC32C-checksummed framing.
- **`pb2/`**: Compiled protobuf classes matching TensorFlow's wire format. To recompile: `protoc --python_out=src --pyi_out=src --proto_path=src src/tfd_utils/pb2/*.proto`
- **`cli.py`**: The `tfd` CLI (`list`, `extract`, `get` subcommands) for TFRecord inspection.

### Index Caching (both readers)

- Built on first access, saved to disk as a pickle file
- Validity checked against file mtimes (all files if ≤5, first 5 + parent dir mtime if >5)
- Parallelized via `ProcessPoolExecutor` when multiple files are provided
- Worker functions (`_process_single_tfrecord`, `_process_single_tar`) are module-level for picklability

### Compressed Tar Performance Note

For compressed tars (gzip etc.), `TarFilePool` keeps `TarFile` objects open and uses `tarfile.extractfile()` with a reconstructed `TarInfo`. Seeking within a `GzipFile` is O(position) for backward seeks — sequential or forward access patterns are fastest.

### TensorFlow Compatibility Tests

- `test_tf_writer_compatibility.py`: files written by `tfd_utils` are readable by TensorFlow
- `test_pb2_compatibility.py`: files written by TensorFlow are readable by `tfd_utils`

These require `tensorflow` (installed via `--extra dev`).

## Dependencies

**Core**: `numpy`, `protobuf`, `crc32c`
**Dev/test only**: `tensorflow`, `pytest`, `Pillow`, `requests`
