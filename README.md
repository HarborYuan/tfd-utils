# TFD Utils

A lightweight Python library for efficient TensorFlow TFRecord processing with random access support. (not requiring TensorFlow)

## 🚀 Key Features

- **🔄 Full TensorFlow Compatibility**: Write with `tfd_utils`, read with TensorFlow (or vice versa) - 100% compatible
- **⚡ Random Access Support**: Access any record by key in O(1) time without reading the entire file
- **🪶 Lightweight & Standalone**: No TensorFlow installation required - works with just `numpy`, `protobuf`, and `crc32c`
- **📦 Ready to Use**: Simple API, automatic index caching, and zero configuration
- **🗂️ Multiple File Support**: Handle single files, lists of files, or glob patterns seamlessly
- **💾 Memory Efficient**: Only loads requested records into memory, not the entire dataset

## Installation

Install via pip (lightweight, no TensorFlow dependency):

```bash
pip install tfd-utils
```

Or for development with optional TensorFlow support:

```bash
# Clone and install
git clone https://github.com/HarborYuan/tfd-utils.git
cd tfd-utils
pip install -e ".[dev]"
```

## Why TFD Utils?

### ✅ **TensorFlow Compatible, TensorFlow Optional**
- Write TFRecords with `tfd_utils`, read with `tf.data.TFRecordDataset` ✅
- Write with `tf.io.TFRecordWriter`, read with `tfd_utils` ✅
- **No TensorFlow installation required** for basic usage

### ✅ **Random Access Made Simple**
- Traditional: Read entire file sequentially to find one record
- TFD Utils: Jump directly to any record by key in O(1) time

### ✅ **Production Ready**
- Automatic index caching for performance
- Robust error handling
- Memory efficient design
- Tested compatibility with TensorFlow 2.x

## Quick Start

### Writing TFRecords (TensorFlow Compatible)

```python
from tfd_utils.writer import TFRecordWriter
from tfd_utils.pb2 import Example, Features, Feature, BytesList

# Create TFRecord files that TensorFlow can read
with TFRecordWriter("data.tfrecord") as writer:
    example = Example(features=Features(feature={
        'key': Feature(bytes_list=BytesList(value=[b'record_1'])),
        'image': Feature(bytes_list=BytesList(value=[image_bytes])),
        'label': Feature(bytes_list=BytesList(value=[b'cat']))
    }))
    writer.write(example.SerializeToString())
```

### Random Access Reading

```python
from tfd_utils.random_access import TFRecordRandomAccess

# Initialize with a single file
reader = TFRecordRandomAccess("data.tfrecord")

# Or with multiple files/patterns
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

### TensorFlow Interoperability

```python
# Read tfd_utils files with TensorFlow
import tensorflow as tf

dataset = tf.data.TFRecordDataset("data.tfrecord")
for record in dataset:
    example = tf.train.Example()
    example.ParseFromString(record.numpy())
    # Process as usual...
```

## Advanced Usage

### Custom Key Feature

```python
# Use different feature as the key (default is 'key')
reader = TFRecordRandomAccess("file.tfrecord", key_feature_name="id")
```

### Custom Index Caching

```python
# Specify custom index location
reader = TFRecordRandomAccess(
    "file.tfrecord",
    index_file="my_custom_index.cache"
)

# Force rebuild index if data changes (usually not needed)
reader.rebuild_index()
```

## License

MIT License

---

**Ready to use, lightweight, and fully TensorFlow compatible!** 🚀
