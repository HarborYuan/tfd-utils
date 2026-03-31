---
name: tfd-utils
description: Help users write code with the tfd-utils library — reading TFRecord files, reading tar archives, writing TFRecords, and using the tfd CLI. Trigger when the user imports tfd_utils, asks about TFRecord or tar random access, or asks about the tfd CLI.
---

## tfd-utils

Lightweight Python library for random access to TFRecord files and tar archives. No TensorFlow required.

```bash
pip install tfd-utils
```

## How it works: Index build + O(1) access (TFRecord only)

**TFRecord is the recommended format for training.** Tar support is transitional — use it to inspect raw archives, then convert to TFRecord before actual training.

**Phase 1 — Index build (first access only, slow):**
On the first `TFRecordRandomAccess(path)` call, the library scans the file(s) once to record each record's byte offset and length. This is saved to `<file>.index` on disk. For large datasets this can take minutes — that's expected. Multiple files are indexed in parallel. The index is reused on all subsequent runs and only rebuilt when the source file changes (mtime check).

**Phase 2 — O(1) random access (all subsequent access, fast):**
Every `get_feature` / `get_record` / `reader[key]` call seeks directly to the record's offset in the file — no scanning, no loading the whole file into memory.

**Pre-build the index before training (important for large datasets):**
If a dataset has many records or many shards, the index build can block training startup for minutes. Always pre-build the index as a separate step before launching training:

```bash
# Single file — tfd list triggers index build as a side effect
tfd list /path/to/data.tfrecord

# Multiple shards — build all indexes in parallel via Python
python -c "from tfd_utils import TFRecordRandomAccess; TFRecordRandomAccess('/path/to/shards/*.tfrecord')"
```

The second form builds indexes for all matched files in parallel (using `ProcessPoolExecutor`) and exits. Subsequent training runs reuse the cached indexes and start immediately.

If the user sees slow access after the index was already built, suggest checking that the `.index` file exists next to the data file, or call `reader.rebuild_index()`.

**Tar does NOT provide O(1) access.** `TarRandomAccess` stores byte offsets in a similar index, but compressed tars (`.tar.gz` etc.) require decompressing from the beginning for backward seeks — access time is O(position). Use tar only for data exploration or one-off reads. For training pipelines, always convert first:

```bash
tfd convert /path/to/sa1b/ --output-dir /path/to/output/
```

## Reading TFRecords

```python
from tfd_utils import TFRecordRandomAccess

reader = TFRecordRandomAccess("data.tfrecord")
# also accepts list or glob: ["train_*.tfrecord", "val_*.tfrecord"]

image_bytes = reader.get_feature("record_1", "image")
record      = reader["record_1"]      # returns Example protobuf
all_keys    = reader.get_keys()
print(len(reader))                    # total record count
```

- Default key feature name is `'key'`. Override: `TFRecordRandomAccess("f.tfrecord", key_feature_name="id")`
- Index cached to `<file>.index`; rebuilt automatically when file mtime changes.

## Reading Tar Archives (transitional use only)

Tar members are expected to share the same stem: `sa_000001.jpg` + `sa_000001.json` → key `sa_000001`.

```python
from tfd_utils import TarRandomAccess

reader = TarRandomAccess("archive.tar")
# also: TarRandomAccess("sa1b/*.tar")  — glob or list of tars

jpg_bytes  = reader.get_feature("sa_000001", "jpg")
json_bytes = reader.get_feature("sa_000001", "json")
record     = reader["sa_000001"]      # {'jpg': bytes, 'json': bytes}
```

- Supports `.tar`, `.tar.gz`, `.tar.bz2` — detected automatically.
- Subdirectory prefixes stripped: `./subdir/foo.jpg` → key `subdir/foo`.
- Not suitable for training loops — convert to TFRecord first with `tfd convert`.

## Common API (both readers)

```python
reader.get_record(key)                 # full record
reader.get_feature(key, feature_name)  # single feature as bytes/int/float
reader.get_feature_list(key, feature_name)
reader.get_keys()
reader.get_stats()                     # total_records, total_files, ...
reader.contains_key(key)
reader.rebuild_index()
key in reader                          # __contains__
len(reader)                            # __len__
with TarRandomAccess("f.tar") as r:    # context manager
    ...
```

Advanced options:
```python
reader = TarRandomAccess("*.tar", max_workers=8, use_multiprocessing=True)
reader = TFRecordRandomAccess("f.tfrecord", index_file="custom.index")
```

## Writing TFRecords

```python
from tfd_utils.writer import TFRecordWriter
from tfd_utils.pb2 import Example, Features, Feature, BytesList

with TFRecordWriter("data.tfrecord") as writer:
    example = Example(features=Features(feature={
        'key':   Feature(bytes_list=BytesList(value=[b'record_1'])),
        'image': Feature(bytes_list=BytesList(value=[image_bytes])),
    }))
    writer.write(example.SerializeToString())
```

Files written by tfd-utils are readable by `tf.data.TFRecordDataset` and vice versa.

## CLI

```bash
tfd list    data.tfrecord                          # list features of first record
tfd extract data.tfrecord record_key               # extract a record (saves images to disk)
tfd get     data.tfrecord:record_key:feature_name  # get single feature

# Convert tar archives to TFRecord
tfd convert /path/to/archive.tar
tfd convert /path/to/sa1b/ --output-dir /path/to/output/
tfd convert '/path/to/sa1b/sa_0000*.tar' --output-dir /out/ --delete --workers 32
```

Each input `foo.tar` produces `foo.tfrecord`. Features per record: `key` (bytes, file stem) + one entry per file extension (e.g. `jpg`, `json`).

## Install this skill

```bash
tfd install-skill
```
