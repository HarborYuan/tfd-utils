# TensorFlow TFRecord Utils

A lightweight Python library for efficient TensorFlow TFRecord processing with random access support, without requiring TensorFlow.

## Key Features

-   **Full TensorFlow Compatibility**: Write with `tfd_utils`, read with TensorFlow, or vice versa. 100% compatible and verified in tests.
-   **Random Access Support**: Access any record by key in O(1) time without reading the entire file.
-   **Lightweight & Standalone**: No TensorFlow installation required. Works with just `numpy`, `protobuf`, and `crc32c`.
-   **Simple API**: Ready to use with automatic index caching and zero configuration.
-   **Multiple File Support**: Handle single files, lists of files, or glob patterns seamlessly.
-   **Memory Efficient**: Only loads requested records into memory, not the entire dataset.

## Installation

Install via pip:

```bash
pip install tfd-utils
```

Or for development with optional TensorFlow support:

```bash
git clone https://github.com/HarborYuan/tfd-utils.git
cd tfd-utils
pip install -e ".[dev]"
```

## Usage

### Writing TFRecords

Create TFRecord files that TensorFlow can read:

```python
from tfd_utils.writer import TFRecordWriter
from tfd_utils.pb2 import Example, Features, Feature, BytesList

with TFRecordWriter("data.tfrecord") as writer:
    example = Example(features=Features(feature={
        'key': Feature(bytes_list=BytesList(value=[b'record_1'])),
        'image': Feature(bytes_list=BytesList(value=[b'your_image_bytes'])),
        'label': Feature(bytes_list=BytesList(value=[b'cat']))
    }))
    writer.write(example.SerializeToString())
```

### Random Access Reading

Initialize with a single file, or with multiple files/patterns:

```python
from tfd_utils.random_access import TFRecordRandomAccess

# Single file
reader = TFRecordRandomAccess("data.tfrecord")

# Multiple files/patterns
reader = TFRecordRandomAccess([
    "train_*.tfrecord",
    "validation_*.tfrecord"
])

# Access any record instantly by key
record = reader.get_record("record_1")
image_bytes = reader.get_feature("record_1", "image")

# Dictionary-like access
if "record_1" in reader:
    record = reader["record_1"]

# Get statistics
print(f"Total records: {len(reader)}")
```

## Command-Line Interface (CLI)

`tfd-utils` comes with a handy command-line tool, `tfd`, for quick inspection of TFRecord files.

### Listing Keys

To list all keys in one or more TFRecord files:

```bash
tfd list /path/to/your/data.tfrecord
```

You can also use glob patterns:

```bash
tfd list 'data_part_*.tfrecord'
```

### Extracting Records

To extract a single record by its key:

```bash
tfd extract /path/to/your/data.tfrecord your_record_key
```

The tool will attempt to automatically detect the content type:
-   **Images** (JPEG, PNG, GIF) are saved to a file (e.g., `your_record_key_image_0.jpeg`).
-   **Text** is printed to the console.
-   Other binary or numerical data is displayed in a readable format.

### TensorFlow Interoperability

Read `tfd_utils` files with TensorFlow:

```python
import tensorflow as tf

dataset = tf.data.TFRecordDataset("data.tfrecord")
for record in dataset:
    example = tf.train.Example()
    example.ParseFromString(record.numpy())
    # Process as usual...
```

### Advanced Usage

#### Custom Key Feature

Use a different feature as the key (default is 'key'):

```python
reader = TFRecordRandomAccess("file.tfrecord", key_feature_name="id")
```

#### Custom Index Caching

Specify a custom index location:

```python
reader = TFRecordRandomAccess(
    "file.tfrecord",
    index_file="my_custom_index.cache"
)

# Force rebuild index if data changes (usually not needed)
reader.rebuild_index()
```

## License

MIT License
