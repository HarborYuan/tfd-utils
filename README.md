# tfd-utils

A lightweight Python library for efficient random access to TensorFlow TFRecord files and tar archives, without requiring TensorFlow.

## Key Features

-   **Unified API**: Access TFRecord files and tar archives through the same interface.
-   **Random Access**: Access any record by key in O(1) time without reading the entire file.
-   **Automatic Index Caching**: Index is built once and cached to disk; rebuilt automatically when files change.
-   **Lightweight & Standalone**: TFRecord support requires only `numpy`, `protobuf`, and `crc32c`. No TensorFlow needed.
-   **Full TensorFlow Compatibility**: Write with `tfd_utils`, read with TensorFlow (or vice versa). 100% compatible.
-   **Multiple File Support**: Single files, lists of files, or glob patterns.
-   **Tar-to-TFRecord Conversion**: CLI tool to batch-convert tar archives to TFRecord format with parallel workers and optional source deletion.
-   **Claude Code Skill**: One-command install (`tfd install-skill`) to enable AI assistance with the library in any project.

## Installation

```bash
pip install tfd-utils
```

## Usage

### TFRecord Random Access

```python
from tfd_utils import TFRecordRandomAccess

reader = TFRecordRandomAccess("data.tfrecord")
# or multiple files / glob patterns
reader = TFRecordRandomAccess(["train_*.tfrecord", "val_*.tfrecord"])

image_bytes = reader.get_feature("record_1", "image")
record = reader["record_1"]   # all features as Example protobuf
print(f"Total records: {len(reader)}")
```

### Tar Archive Random Access

Tar archives are expected to contain paired files sharing the same stem:

```
sa_000001.jpg   →  key='sa_000001', feature='jpg'
sa_000001.json  →  key='sa_000001', feature='json'
```

Both uncompressed (`.tar`) and compressed (`.tar.gz`, etc.) archives are supported.

```python
from tfd_utils import TarRandomAccess

reader = TarRandomAccess("archive.tar")
# or glob / list of tars
reader = TarRandomAccess("sa1b/*.tar")

jpg_bytes  = reader.get_feature("sa_000001", "jpg")
json_bytes = reader.get_feature("sa_000001", "json")
record     = reader["sa_000001"]   # {'jpg': bytes, 'json': bytes}
print(f"Total records: {len(reader)}")
```

Member paths with subdirectory prefixes are handled automatically:
`./subdir/foo.jpg` → key `subdir/foo`.

#### Example: SA-1B Dataset

SA-1B tars are gzip-compressed and contain paired `.jpg` / `.json` files per image:

```python
import json
from PIL import Image
import io
from tfd_utils import TarRandomAccess

# Point to one or more SA-1B tar files (compressed tars are supported)
reader = TarRandomAccess("/path/to/sa1b/sa_000020.tar")
# or load multiple shards at once
reader = TarRandomAccess("/path/to/sa1b/*.tar")

# Each key is the image ID (e.g. 'sa_226692')
keys = reader.get_keys()
print(f"Images in this shard: {len(keys)}")

key = keys[0]

# Load the JPEG image
jpg_bytes = reader.get_feature(key, "jpg")
image = Image.open(io.BytesIO(jpg_bytes))

# Load the annotation (segmentation masks, bounding boxes, …)
json_bytes = reader.get_feature(key, "json")
annotation = json.loads(json_bytes)
print(f"Image size : {annotation['image']['width']}x{annotation['image']['height']}")
print(f"Masks      : {len(annotation['annotations'])}")
```

### Writing TFRecords

```python
from tfd_utils.writer import TFRecordWriter
from tfd_utils.pb2 import Example, Features, Feature, BytesList

with TFRecordWriter("data.tfrecord") as writer:
    example = Example(features=Features(feature={
        'key':   Feature(bytes_list=BytesList(value=[b'record_1'])),
        'image': Feature(bytes_list=BytesList(value=[b'<image bytes>'])),
    }))
    writer.write(example.SerializeToString())
```

### Common API (both readers)

```python
reader.get_record(key)                    # full record
reader.get_feature(key, feature_name)     # single feature
reader.get_feature_list(key, feature_name)
reader.get_keys()                         # all keys
reader.get_stats()                        # total_records, total_files, ...
reader.contains_key(key)
reader.rebuild_index()

key in reader                             # __contains__
reader[key]                               # __getitem__ (raises KeyError if missing)
len(reader)                               # __len__

with TarRandomAccess("archive.tar") as r: # context manager
    ...
```

### Advanced Options

```python
# TFRecord: custom key feature name (default 'key')
reader = TFRecordRandomAccess("file.tfrecord", key_feature_name="id")

# Both: custom index file location
reader = TFRecordRandomAccess("file.tfrecord", index_file="my.index")
reader = TarRandomAccess("archive.tar", index_file="my.tar_index")

# Both: control parallelism
reader = TarRandomAccess("*.tar", max_workers=8, use_multiprocessing=True)
```

## CLI

```bash
tfd list    /path/to/data.tfrecord
tfd extract /path/to/data.tfrecord record_key
tfd get     /path/to/data.tfrecord:record_key:feature_name
```

### Claude Code Skill

Install the tfd-utils skill for Claude Code to get AI assistance with the library in any project:

```bash
tfd install-skill
```

This copies a skill file to `~/.claude/skills/tfd-utils/SKILL.md`, enabling Claude to assist with TFRecord and tar access, the CLI, and best practices.

### Converting Tar Archives to TFRecord

The `tfd convert` command converts tar archive(s) to TFRecord files. Each record stores all
file extensions as bytes features plus a `key` feature containing the file stem.

```bash
# Convert a single tar
tfd convert /path/to/archive.tar

# Convert all tars in a directory, write to a different output directory
tfd convert /path/to/sa1b/ --output-dir /path/to/output/

# Glob pattern
tfd convert '/path/to/sa1b/sa_0000*.tar' --output-dir /path/to/output/

# Delete each source tar after successful conversion
tfd convert /path/to/sa1b/ --output-dir /path/to/output/ --delete

# Control parallelism (default: 16 workers)
tfd convert /path/to/sa1b/ --output-dir /path/to/output/ --workers 32
```

Each input `foo.tar` produces `foo.tfrecord` in the output directory (default: same directory
as the source). A TFRecord produced from SA-1B tars contains these features per record:

| Feature | Type  | Content                          |
|---------|-------|----------------------------------|
| `key`   | bytes | File stem, e.g. `sa_226692`      |
| `jpg`   | bytes | Raw JPEG image bytes             |
| `json`  | bytes | Annotation JSON (masks, boxes…)  |

```python
from tfd_utils import TFRecordRandomAccess
import json

reader = TFRecordRandomAccess("/path/to/output/sa_000000.tfrecord")
jpg_bytes  = reader.get_feature("sa_226692", "jpg")
json_bytes = reader.get_feature("sa_226692", "json")
annotation = json.loads(json_bytes)
```

## TensorFlow Interoperability

```python
import tensorflow as tf

dataset = tf.data.TFRecordDataset("data.tfrecord")
for record in dataset:
    example = tf.train.Example()
    example.ParseFromString(record.numpy())
```

## License

MIT License
