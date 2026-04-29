# tfd-utils

Lightweight Python library for **O(1) random access** to TensorFlow TFRecord files and tar archives. No TensorFlow dependency required for the core library.

- Unified API for TFRecord and tar (read by key in O(1))
- Index built once and cached to disk; rebuilt automatically on file change
- 100% wire-compatible with `tf.data.TFRecordDataset` (read either direction)
- Multi-file / glob support, parallel index build
- `tfd` CLI: `list`, `extract`, `get`, `convert`, `prebuild`, `install-skill`

---

## Installation

```bash
pip install tfd-utils
```

Core dependencies: `numpy`, `protobuf`, `crc32c`. TensorFlow is **not** required.

---

## Quickstart

### Read a TFRecord

```python
from tfd_utils import TFRecordRandomAccess

reader = TFRecordRandomAccess("data.tfrecord")
# also accepts a list or glob: ["train_*.tfrecord", "val_*.tfrecord"]

image_bytes = reader.get_feature("record_1", "image")
record      = reader["record_1"]      # Example protobuf
print(len(reader), "records")
```

### Write a TFRecord

```python
from tfd_utils.writer import TFRecordWriter
from tfd_utils.pb2 import Example, Features, Feature, BytesList

with TFRecordWriter("data.tfrecord") as w:
    ex = Example(features=Features(feature={
        'key':   Feature(bytes_list=BytesList(value=[b'record_1'])),
        'image': Feature(bytes_list=BytesList(value=[image_bytes])),
    }))
    w.write(ex.SerializeToString())
```

### Read a tar archive

Tar members sharing a stem are grouped under the same key:
`sa_000001.jpg` + `sa_000001.json` → key `sa_000001`, features `jpg` / `json`.

```python
from tfd_utils import TarRandomAccess

reader = TarRandomAccess("archive.tar")          # also: "sa1b/*.tar"
jpg_bytes  = reader.get_feature("sa_000001", "jpg")
json_bytes = reader.get_feature("sa_000001", "json")
record     = reader["sa_000001"]                  # {'jpg': bytes, 'json': bytes}
```

`.tar`, `.tar.gz`, and `.tar.bz2` are supported (autodetected). Tar is not O(1) — for training pipelines, **convert to TFRecord first** (see [Converting tar → TFRecord](#converting-tar--tfrecord)).

---

## Pre-build the index for large datasets

The first call to `TFRecordRandomAccess(...)` scans every shard to record byte offsets, then caches a `<file>.index` next to each shard. For thousands of shards or hundreds of millions of records, this **first-time scan can take minutes to hours**.

**Do not let your training script trigger the index build.** Symptoms when you do:

- The job appears to hang with no progress, holding GPUs while doing pure CPU/IO work.
- Multi-rank launchers (`torchrun`, `accelerate`, …) race to build the *same* index from every rank, multiplying cost.

**Recommended workflow** — pre-build once on a CPU/login node, then launch training:

```bash
# Default — builds .index for every matching shard in parallel
tfd prebuild '/path/to/shards/*.tfrecord'

# Bump concurrency on a fat CPU node (default is 2x CPU count, min 32)
tfd prebuild '/path/to/shards/*.tfrecord' --workers 128
```

Verify the indexes exist before submitting the training job:

```bash
ls /path/to/shards/*.index | wc -l   # should equal shard count
```

Subsequent runs reuse the cached `.index` files (mtime-checked) and start instantly.

> **Programmatic equivalent** (only if you cannot run the CLI):
> ```python
> from tfd_utils import TFRecordRandomAccess
> TFRecordRandomAccess('/path/to/shards/*.tfrecord', max_workers=128)
> ```

---

## CLI

```bash
tfd list     data.tfrecord                          # show features of the first record
tfd extract  data.tfrecord <key>                    # extract a record (saves images to disk)
tfd get      data.tfrecord:<key>:<feature>          # extract a single feature

tfd prebuild '/path/to/shards/*.tfrecord'           # build .index ahead of training
tfd prebuild '/path/to/shards/*.tfrecord' -w 128

tfd convert  /path/to/sa1b/ -o /out/                # tar(s) → TFRecord(s)
tfd convert  '/path/to/sa1b/sa_*.tar' -o /out/ -d -w 32

tfd install-skill                                   # install the Claude Code skill
```

### Converting tar → TFRecord

`tfd convert` reads each input tar and writes one TFRecord per source file. Each output record contains a `key` feature (file stem) plus one bytes feature per file extension.

```bash
tfd convert /path/to/archive.tar                                  # single tar
tfd convert /path/to/sa1b/ --output-dir /out/                     # directory of tars
tfd convert '/path/to/sa1b/sa_0000*.tar' --output-dir /out/       # glob
tfd convert /path/to/sa1b/ --output-dir /out/ --delete            # delete sources on success
tfd convert /path/to/sa1b/ --output-dir /out/ --workers 32        # default is 16 workers
```

For SA-1B-style tars (paired `.jpg` + `.json` per image), each output record has:

| Feature | Type  | Content                          |
|---------|-------|----------------------------------|
| `key`   | bytes | File stem, e.g. `sa_226692`      |
| `jpg`   | bytes | Raw JPEG image bytes             |
| `json`  | bytes | Annotation JSON (masks, boxes…)  |

---

## API reference

### Common API (both readers)

```python
reader.get_record(key)                 # full record
reader.get_feature(key, feature_name)  # single feature (bytes / int / float)
reader.get_feature_list(key, feature_name)
reader.get_keys()                      # all keys
reader.get_stats()                     # {'total_records': ..., 'total_files': ..., ...}
reader.contains_key(key)
reader.rebuild_index()                 # force rebuild

key in reader                          # __contains__
reader[key]                            # __getitem__ (raises KeyError if missing)
len(reader)                            # __len__

with TarRandomAccess("archive.tar") as r:   # context manager
    ...
```

### Constructor options

```python
# TFRecord: custom key feature name (default 'key')
TFRecordRandomAccess("file.tfrecord", key_feature_name="id")

# Both: custom index file location
TFRecordRandomAccess("file.tfrecord", index_file="my.index")
TarRandomAccess("archive.tar", index_file="my.tar_index")

# Both: control parallelism for index build
TFRecordRandomAccess("*.tfrecord", max_workers=128)
TarRandomAccess("*.tar", max_workers=8, use_multiprocessing=True)
```

### Example: SA-1B

```python
import json, io
from PIL import Image
from tfd_utils import TarRandomAccess

reader = TarRandomAccess("/path/to/sa1b/*.tar")    # gzip tars supported
key = reader.get_keys()[0]                          # e.g. 'sa_226692'

image      = Image.open(io.BytesIO(reader.get_feature(key, "jpg")))
annotation = json.loads(reader.get_feature(key, "json"))
print(f"{annotation['image']['width']}x{annotation['image']['height']},",
      f"{len(annotation['annotations'])} masks")
```

---

## TensorFlow interoperability

Files written by `tfd_utils` are byte-identical to TensorFlow's TFRecord format:

```python
import tensorflow as tf
for record in tf.data.TFRecordDataset("data.tfrecord"):
    ex = tf.train.Example()
    ex.ParseFromString(record.numpy())
```

The reverse direction works too — `TFRecordRandomAccess` reads files written by `tf.io.TFRecordWriter`.

---

## Claude Code skill

Install the bundled skill so [Claude Code](https://claude.com/claude-code) can assist with the library in any project:

```bash
tfd install-skill
```

Copies a versioned `SKILL.md` to `~/.claude/skills/tfd-utils/`. Re-run after upgrading the library to refresh the skill content:

```bash
pip install -U tfd-utils && tfd install-skill
```

---

## Version notes

### v1.0.0

- New `tfd prebuild` CLI command — pre-build `.index` files for one or more shards before training, with high default concurrency (`max(2x CPU, 32)` workers, override with `-w`). Avoids the multi-rank race + multi-minute training-startup hang on large datasets.
- `tfd_utils.__version__` is now exposed (sourced from package metadata).
- `tfd install-skill` stamps the installed library version into the bundled Claude Code skill.
- README rewritten for clarity; pre-build workflow promoted to a top-level section.

### v0.4.3 — concurrency fix (recommended upgrade)

Versions before 0.4.3 have a critical concurrency bug: when multiple processes built the index simultaneously, they could corrupt the `.index` file, causing `_pickle.UnpicklingError` on the next run.

v0.4.3 introduces:

- **Exclusive build lock** — only one process builds the index at a time; others wait and reuse the result.
- **Atomic index write** — index is written to `.tmp` and renamed into place; a killed process can never leave a half-written index behind.

If you hit `_pickle.UnpicklingError` on an existing dataset, delete the stale `.index` files and upgrade:

```bash
rm /path/to/data/*.index
pip install --upgrade tfd-utils
```

---

## License

MIT
