# TFD Utils

A Python library for efficient TensorFlow TFRecord processing and random access.

## Features

- **Random Access to TFRecord Files**: Efficiently access specific records in TFRecord files without reading the entire file
- **Automatic Index Caching**: Builds and caches an index on first access for fast subsequent lookups
- **Multiple File Support**: Handle single files, lists of files, or glob patterns
- **Flexible Key Types**: Support for string, integer, and float keys
- **Memory Efficient**: Only loads requested records into memory

## Quick Start

```python
from tfd_utils.random_access import TFRecordRandomAccess

# Initialize with a single file
reader = TFRecordRandomAccess("path/to/your/file.tfrecord")

# Or with multiple files
reader = TFRecordRandomAccess([
    "path/to/file1.tfrecord",
    "path/to/file2.tfrecord"
])

# Or with a glob pattern
reader = TFRecordRandomAccess("path/to/data_*.tfrecord")

# Get a record by key
record = reader.get_record("your_key")

# Get a specific feature from a record
image_bytes = reader.get_feature("your_key", "image")

# Check if key exists
if "your_key" in reader:
    print("Key exists!")

# Get statistics
stats = reader.get_stats()
print(f"Total records: {stats['total_records']}")
```

## Advanced Usage

### Custom Key Feature

By default, the library looks for keys in a feature named 'key'. You can specify a different feature name:

```python
# Use 'id' feature as the key
reader = TFRecordRandomAccess("file.tfrecord", key_feature_name="id")
```

### Custom Index File

You can specify where to save the index cache:

```python
reader = TFRecordRandomAccess(
    "file.tfrecord",
    index_file="my_custom_index.cache"
)
```

### Rebuilding Index

If your TFRecord files change, you can rebuild the index:

```python
reader.rebuild_index()
```

## License

MIT License