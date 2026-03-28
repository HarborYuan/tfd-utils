# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`tfd_utils` is a lightweight Python library for reading and writing TensorFlow TFRecord files with **efficient O(1) random access by key**, without requiring TensorFlow as a dependency. It is 100% compatible with TensorFlow's `tf.data.TFRecordDataset` and `tf.io.TFRecordWriter`.

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

# Build the package
uv build
```

## Architecture

### Core Concepts

- **Index file (`.tfd_index`)**: On first access, `TFRecordRandomAccess` scans TFRecord files and builds a JSON index mapping each record's key to its byte offset. This index is cached to disk for O(1) subsequent lookups. Index validity is checked against file modification times.
- **Key feature**: Each `tf.train.Example` must have a feature named `'key'` (customizable via `key_feature_name` parameter) containing a unique string identifier. This is how records are addressed.
- **Protocol Buffers**: The `pb2/` directory contains compiled protobuf files from TensorFlow's own `.proto` definitions. To recompile: `protoc --python_out=src --pyi_out=src --proto_path=src src/tfd_utils/pb2/*.proto`

### Module Responsibilities

- **`random_access.py`**: `TFRecordRandomAccess` — the main read API. `FileContextPool` manages a thread-safe LRU cache of open file handles (default 100) to avoid descriptor exhaustion. Supports single files, lists, and glob patterns. Index building can be parallelized via multiprocessing.
- **`writer/tf_writer.py`**: `TFRecordWriter` — context manager for writing TFRecord files with proper length-prefixed, CRC32C-checksummed framing.
- **`pb2/`**: Compiled protobuf classes (`Example`, `Features`, `Feature`, etc.) matching TensorFlow's wire format.
- **`cli.py`**: The `tfd` CLI entry point with three subcommands: `list` (features in first record), `extract` (full record by key), `get` (single feature via `path:key:feature` syntax).

### Compatibility Tests

Integration tests in `tests/integration/` verify bidirectional TensorFlow compatibility:
- `test_tf_writer_compatibility.py`: files written by `tfd_utils` are readable by TensorFlow
- `test_pb2_compatibility.py`: files written by TensorFlow are readable by `tfd_utils`

These tests require `tensorflow` (installed via `--extra dev`).

## Dependencies

**Core** (no TensorFlow needed at runtime): `numpy`, `protobuf`, `crc32c`
**Dev/test only**: `tensorflow`, `pytest`, `Pillow`, `requests`
