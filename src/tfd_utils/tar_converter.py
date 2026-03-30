"""
Convert tar archives to TFRecord files.

Reads each tar file sequentially, groups members by file stem, and writes one
TFRecord per tar. Each record stores all file extensions as bytes features plus
a 'key' feature containing the stem.
"""

import glob
import os
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Tuple

from .pb2 import BytesList, Example, Feature, Features
from .writer.tf_writer import TFRecordWriter


def _build_example(key: str, features: Dict[str, bytes]) -> bytes:
    """Serialize a key + feature-dict into a TFRecord Example protobuf."""
    feature_map = {
        'key': Feature(bytes_list=BytesList(value=[key.encode('utf-8')])),
    }
    for ext, data in features.items():
        feature_map[ext] = Feature(bytes_list=BytesList(value=[data]))
    example = Example(features=Features(feature=feature_map))
    return example.SerializeToString()


def convert_tar_to_tfrecord(
    tar_path: str,
    output_path: str,
) -> int:
    """Convert a single tar file to a TFRecord file.

    Reads the tar sequentially, groups members by stem, and writes one Example
    per key. Returns the number of records written.

    Args:
        tar_path: Path to the source tar file.
        output_path: Path to write the output TFRecord file.

    Returns:
        Number of records written.
    """
    # Buffer all members grouped by stem: {stem: {ext: bytes}}
    records: Dict[str, Dict[str, bytes]] = {}

    with tarfile.open(tar_path, 'r:*') as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            p = PurePosixPath(member.name)
            ext = p.suffix.lstrip('.')
            if not ext:
                continue
            key = p.with_suffix('').as_posix()
            # Strip leading './' if present
            if key.startswith('./'):
                key = key[2:]
            f = tf.extractfile(member)
            if f is None:
                continue
            records.setdefault(key, {})[ext] = f.read()

    count = 0
    with TFRecordWriter(output_path) as writer:
        for key, features in records.items():
            writer.write(_build_example(key, features))
            count += 1

    return count


def _convert_one(args: Tuple[str, str, bool]) -> Tuple[str, int]:
    """Worker function for parallel conversion. Returns (tar_path, count) or raises."""
    tar_path, out_path, delete_after = args
    count = convert_tar_to_tfrecord(tar_path, out_path)
    if delete_after:
        os.remove(tar_path)
    return tar_path, count


def convert_tars(
    input_paths: List[str],
    output_dir: Optional[str],
    delete_after: bool,
    workers: int = 16,
) -> None:
    """Convert a list of tar files to TFRecord files.

    Args:
        input_paths: List of tar file paths to convert.
        output_dir: Directory to write TFRecord files. Defaults to same
            directory as each source tar if None.
        delete_after: If True, delete each source tar after successful
            conversion.
        workers: Number of parallel worker processes (default 16).
    """
    jobs: List[Tuple[str, str, bool]] = []
    for tar_path in input_paths:
        tar_path = os.path.abspath(tar_path)
        stem = Path(tar_path).stem
        out_dir = output_dir if output_dir else os.path.dirname(tar_path)
        out_path = os.path.join(out_dir, f"{stem}.tfrecord")
        jobs.append((tar_path, out_path, delete_after))

    failed = []
    completed = 0

    if workers <= 1 or len(jobs) == 1:
        for tar_path, out_path, do_delete in jobs:
            print(f"Converting {os.path.basename(tar_path)} -> {out_path} ...", flush=True)
            try:
                count = convert_tar_to_tfrecord(tar_path, out_path)
                print(f"  Wrote {count} records.", flush=True)
                if do_delete:
                    os.remove(tar_path)
                    print(f"  Deleted {tar_path}", flush=True)
                completed += 1
            except Exception as e:
                print(f"  ERROR: {e}", flush=True)
                failed.append(tar_path)
                if os.path.exists(out_path):
                    try:
                        os.remove(out_path)
                    except Exception:
                        pass
    else:
        print(f"Using {workers} worker processes for {len(jobs)} file(s)...", flush=True)
        out_path_map = {tar_path: out_path for tar_path, out_path, _ in jobs}
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_map = {executor.submit(_convert_one, job): job[0] for job in jobs}
            for future in as_completed(future_map):
                tar_path = future_map[future]
                out_path = out_path_map[tar_path]
                try:
                    _, count = future.result()
                    action = " + deleted" if delete_after else ""
                    print(
                        f"  [{completed + 1}/{len(jobs)}] {os.path.basename(tar_path)}"
                        f" -> {count} records{action}",
                        flush=True,
                    )
                    completed += 1
                except Exception as e:
                    print(f"  ERROR {os.path.basename(tar_path)}: {e}", flush=True)
                    failed.append(tar_path)
                    if os.path.exists(out_path):
                        try:
                            os.remove(out_path)
                        except Exception:
                            pass

    print(f"\nDone: {completed}/{len(jobs)} converted.", flush=True)
    if failed:
        print(f"{len(failed)} file(s) failed:")
        for f in failed:
            print(f"  {f}")


def resolve_input_paths(path_spec: str) -> List[str]:
    """Resolve a path spec (file, directory, or glob) to a sorted list of tar paths."""
    if os.path.isfile(path_spec):
        return [path_spec]
    if os.path.isdir(path_spec):
        return sorted(glob.glob(os.path.join(path_spec, '*.tar')))
    matched = sorted(glob.glob(path_spec))
    return matched
