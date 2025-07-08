# TFRecord Random Access

This module provides the `TFRecordRandomAccess` class for efficient random access to TFRecord files with automatic index caching.

## Features

- **Efficient Random Access**: Build an index once, then access any record by key in O(1) time
- **Automatic Caching**: Index is built on first access and cached for subsequent uses
- **Multiple File Support**: Works with single files, lists of files, or glob patterns
- **Flexible Key Types**: Supports string, integer, and float keys
- **Progress Tracking**: Shows progress during index building
- **Memory Efficient**: Only loads the index, not the entire dataset

## Installation

The package is managed by `uv`. Make sure you have the required dependencies:

```bash
uv sync
```

## Usage

### Basic Usage

```python
from tfd_utils import TFRecordRandomAccess

# Create a random access reader
reader = TFRecordRandomAccess(
    tfrecord_path="/path/to/your/file.tfrecord",
    key_feature_name="key"  # Name of the feature containing the record key
)

# Access a record by key
example = reader.get_record("your_key")

# Get a specific feature from a record
image_bytes = reader.get_feature("your_key", "image")

# Dictionary-like access
example = reader["your_key"]

# Check if key exists
if "your_key" in reader:
    print("Key exists!")
```

### Multiple Files

```python
# Using glob pattern
reader = TFRecordRandomAccess(
    tfrecord_path="/path/to/files/*.tfrecord",
    key_feature_name="key"
)

# Using list of files
reader = TFRecordRandomAccess(
    tfrecord_path=[
        "/path/to/file1.tfrecord",
        "/path/to/file2.tfrecord"
    ],
    key_feature_name="key"
)
```

### Custom Index Location

```python
reader = TFRecordRandomAccess(
    tfrecord_path="/path/to/your/file.tfrecord",
    key_feature_name="key",
    index_file="/custom/path/to/index.pkl"
)
```

### Advanced Usage

```python
# Get statistics
stats = reader.get_stats()
print(f"Total records: {stats['total_records']}")
print(f"Records per file: {stats['records_per_file']}")

# Get all keys
all_keys = reader.get_keys()

# Get raw record bytes
raw_bytes = reader.get_raw_record("your_key")

# Force rebuild index
reader.rebuild_index()

# Get number of records
num_records = len(reader)
```

## API Reference

### TFRecordRandomAccess

#### Constructor

```python
TFRecordRandomAccess(
    tfrecord_path: Union[str, Path, List[str], List[Path]],
    key_feature_name: str = 'key',
    index_file: Optional[Union[str, Path]] = None,
    progress_interval: int = 1000
)
```

**Parameters:**
- `tfrecord_path`: Path to TFRecord file(s). Can be a single file, list of files, or glob pattern.
- `key_feature_name`: Name of the feature containing the record key (default: 'key')
- `index_file`: Optional path to save/load the index cache. Auto-generated if None.
- `progress_interval`: Print progress every N records during indexing (default: 1000)

#### Methods

- `get_record(key: str) -> Optional[tf.train.Example]`: Get a TFRecord by key
- `get_raw_record(key: str) -> Optional[bytes]`: Get raw record bytes by key
- `get_feature(key: str, feature_name: str) -> Optional[Any]`: Get specific feature value
- `contains_key(key: str) -> bool`: Check if key exists
- `get_keys() -> List[str]`: Get all available keys
- `get_stats() -> Dict[str, Any]`: Get statistics about indexed records
- `rebuild_index() -> None`: Force rebuild the index

#### Special Methods

- `len(reader)`: Get number of records
- `key in reader`: Check if key exists
- `reader[key]`: Get record by key (raises KeyError if not found)

## Index File Format

The index is stored as a pickled dictionary with the following structure:

```python
{
    "key1": {
        "file": "/path/to/file.tfrecord",
        "offset": 1234,
        "length": 5678
    },
    "key2": {
        "file": "/path/to/file.tfrecord", 
        "offset": 5912,
        "length": 2345
    }
}
```

## Performance

- **Index Building**: O(n) where n is the total number of records
- **Record Access**: O(1) after index is built
- **Memory Usage**: Only the index is kept in memory (~50-100 bytes per record)

## Examples

See `example_usage.py` and `test_with_experimental_data.py` for complete examples.

## Requirements

- Python >= 3.10
- TensorFlow >= 2.13.0
