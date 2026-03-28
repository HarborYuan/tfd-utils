"""
Tar Archive Random Access Reader

Provides random access to tar archives (compressed or uncompressed) containing
paired files (e.g., xxx.jpg + xxx.json) by building and caching a byte-offset index.

Supports both uncompressed (.tar) and compressed (.tar.gz, .tar.bz2, etc.) archives.
For uncompressed tars, access is O(1) via raw file seeks. For compressed tars, access
requires seeking within the decompressed stream (which may be slow for random access
on large archives — sequential or nearly-sequential access is recommended).
"""

import os
import pickle
import glob
import tarfile
from collections import OrderedDict
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Dict, List, Optional, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from functools import partial


class TarFilePool:
    """
    LRU cache of open TarFile objects to avoid repeated opens during random access.

    Keeps TarFile objects open so that the underlying file handle (and GzipFile state
    for compressed archives) is reused across multiple extractfile() calls.
    """

    def __init__(self, max_size: int = 20):
        self.max_size = max_size
        self._pool: OrderedDict[str, tarfile.TarFile] = OrderedDict()

    def get_tarfile(self, file_path: str) -> tarfile.TarFile:
        """Return an open TarFile for file_path, opening it if necessary."""
        if file_path in self._pool:
            tf = self._pool.pop(file_path)
            self._pool[file_path] = tf
            return tf

        tf = tarfile.open(file_path, 'r:*')

        if len(self._pool) >= self.max_size:
            _, oldest = self._pool.popitem(last=False)
            try:
                oldest.close()
            except Exception:
                pass

        self._pool[file_path] = tf
        return tf

    def close_all(self) -> None:
        """Close all open TarFile handles."""
        for tf in self._pool.values():
            try:
                tf.close()
            except Exception:
                pass
        self._pool.clear()

    def __del__(self) -> None:
        self.close_all()


def _process_single_tar(
    tar_file: str,
    progress_interval: int = 1024,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Process a single tar file and return its index.

    Index structure:
        {key: {ext: {'file': str, 'member_name': str, 'data_offset': int, 'size': int}}}

    Key is derived from the member name stem (preserving subdirectory structure).
    Extension is the member name suffix without the leading dot.
    Members without an extension or that are not regular files are skipped.

    Supports both uncompressed ('r:') and compressed ('r:*') archives via auto-detection.
    """
    print(f"Processing {os.path.basename(tar_file)}...")

    index: Dict[str, Dict[str, Dict[str, Any]]] = {}
    count = 0

    with tarfile.open(tar_file, 'r:*') as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue

            p = PurePosixPath(member.name)
            ext = p.suffix.lstrip('.')
            if not ext:
                continue

            key = p.with_suffix('').as_posix()

            if key not in index:
                index[key] = {}

            index[key][ext] = {
                'file': tar_file,
                'member_name': member.name,
                'data_offset': member.offset_data,
                'size': member.size,
            }

            count += 1
            if count % progress_interval == 0:
                print(f"  Processed {count} members from {os.path.basename(tar_file)}")

    print(f"  Completed {os.path.basename(tar_file)}: {len(index)} keys, {count} members")
    return index


class TarRandomAccess:
    """
    Random access reader for tar archives (compressed or uncompressed).

    Archives are expected to contain paired files sharing the same stem, e.g.:
        xxx.jpg   (key='xxx', feature='jpg')
        xxx.json  (key='xxx', feature='json')

    Member paths with subdirectories are supported; the key includes the directory
    prefix (e.g., member 'subdir/foo.jpg' → key 'subdir/foo').

    On first access an index is built mapping each key to the byte offsets of its
    members inside the tar file(s). The index is cached to disk so subsequent
    instantiations skip the scanning step.

    Both uncompressed and compressed (gzip, bzip2, etc.) archives are supported.
    For compressed archives, performance may degrade for non-sequential access
    patterns since seeking backwards within the compressed stream requires
    re-reading from the start.
    """

    def __init__(
        self,
        tar_path: Union[str, Path, List[str], List[Path]],
        index_file: Optional[Union[str, Path]] = None,
        progress_interval: int = 1024,
        max_workers: Optional[int] = None,
        use_multiprocessing: bool = True,
        file_pool_size: int = 20,
    ):
        """
        Initialize the tar random access reader.

        Args:
            tar_path: Path to tar file(s). Can be:
                - Single file path (str or Path)
                - List of file paths
                - Glob pattern (str) for multiple files
            index_file: Optional path for the index cache. Auto-generated if None.
            progress_interval: Print progress every N members during indexing.
            max_workers: Worker processes for parallel indexing. Defaults to CPU count.
            use_multiprocessing: Enable parallel indexing for multiple files.
            file_pool_size: Maximum open TarFile handles kept in the LRU pool.
        """
        self.progress_interval = progress_interval
        self.max_workers = max_workers or multiprocessing.cpu_count()

        self.tarfile_pool = TarFilePool(max_size=file_pool_size)

        self.tar_files = self._resolve_tar_files(tar_path)
        if not self.tar_files:
            raise ValueError(f"No tar files found for path: {tar_path}")

        self.use_multiprocessing = use_multiprocessing and len(self.tar_files) > 1

        self.index_file = self._get_index_file_path(index_file)

        self._index: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None

    def _resolve_tar_files(
        self, tar_path: Union[str, Path, List[str], List[Path]]
    ) -> List[str]:
        """Resolve the input path(s) to a sorted list of tar file paths."""
        if isinstance(tar_path, (list, tuple)):
            files = []
            for path in tar_path:
                path_str = str(path)
                if os.path.exists(path_str):
                    files.append(path_str)
                else:
                    files.extend(glob.glob(path_str))
            return sorted(files)
        else:
            path_str = str(tar_path)
            if os.path.exists(path_str):
                return [path_str]
            else:
                return sorted(glob.glob(path_str))

    def _get_index_file_path(self, index_file: Optional[Union[str, Path]]) -> str:
        """Generate index file path if not provided."""
        if index_file is not None:
            return str(index_file)

        first_file = Path(self.tar_files[0])
        if len(self.tar_files) == 1:
            return str(first_file.with_suffix('.tar_index'))
        else:
            return str(first_file.parent / f"{first_file.stem}_unified.tar_index")

    def _is_index_valid(self) -> bool:
        """Check if the cached index is still valid."""
        if not os.path.exists(self.index_file):
            return False

        index_mtime = os.path.getmtime(self.index_file)

        if len(self.tar_files) > 5:
            parent_dir = os.path.dirname(self.tar_files[0])
            parent_mtime = os.path.getmtime(parent_dir)
            if parent_mtime > index_mtime + 0.5:
                return False
            files_to_check = self.tar_files[:5]
        else:
            files_to_check = self.tar_files

        for tar_file in files_to_check:
            if not os.path.exists(tar_file):
                return False
            if os.path.getmtime(tar_file) > index_mtime:
                return False

        return True

    def _build_index(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Build index for all tar files."""
        print(f"Building index for {len(self.tar_files)} tar file(s)...")

        if self.use_multiprocessing and len(self.tar_files) > 1:
            return self._build_index_parallel()
        else:
            return self._build_index_sequential()

    def _build_index_sequential(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Build index sequentially."""
        index: Dict[str, Dict[str, Dict[str, Any]]] = {}
        total_keys = 0

        for tar_file in self.tar_files:
            file_index = _process_single_tar(tar_file, self.progress_interval)
            index.update(file_index)
            total_keys += len(file_index)

        print(f"Total keys indexed: {total_keys}")

        with open(self.index_file, 'wb') as f:
            pickle.dump(index, f)
        print(f"Index saved to {self.index_file}")

        return index

    def _build_index_parallel(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Build index using multiprocessing."""
        print(f"Using {self.max_workers} worker processes for parallel indexing...")

        index: Dict[str, Dict[str, Dict[str, Any]]] = {}
        total_keys = 0

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            process_func = partial(
                _process_single_tar,
                progress_interval=self.progress_interval,
            )
            future_to_file = {
                executor.submit(process_func, tar_file): tar_file
                for tar_file in self.tar_files
            }

            for future in as_completed(future_to_file):
                tar_file = future_to_file[future]
                try:
                    file_index = future.result()
                    index.update(file_index)
                    total_keys += len(file_index)
                except Exception as e:
                    print(f"Error processing {tar_file}: {e}")

        print(f"Total keys indexed: {total_keys}")

        with open(self.index_file, 'wb') as f:
            pickle.dump(index, f)
        print(f"Index saved to {self.index_file}")

        return index

    def _load_index(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Load index from cache or build if absent/stale."""
        if self._is_index_valid():
            print(f"Loading index from {self.index_file}")
            with open(self.index_file, 'rb') as f:
                return pickle.load(f)
        else:
            print("Index cache is invalid or missing, rebuilding...")
            return self._build_index()

    @property
    def index(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Lazy-loaded index: {key: {ext: {file, member_name, data_offset, size}}}."""
        if self._index is None:
            self._index = self._load_index()
        return self._index

    def _read_member(self, info: Dict[str, Any]) -> bytes:
        """Read member data using TarFile.extractfile() with stored offset info."""
        tf = self.tarfile_pool.get_tarfile(info['file'])

        # Reconstruct a minimal TarInfo so we can call extractfile() without
        # needing to pre-load all members via getmembers().
        tar_info = tarfile.TarInfo(name=info['member_name'])
        tar_info.offset_data = info['data_offset']
        tar_info.size = info['size']

        ef = tf.extractfile(tar_info)
        return ef.read()

    def get_record(self, key: str) -> Optional[Dict[str, bytes]]:
        """
        Get all features for a key as a dict mapping extension to raw bytes.

        Args:
            key: The record key (file stem, preserving subdirectory prefix).

        Returns:
            Dict mapping feature name (extension) to bytes, or None if not found.
        """
        if key not in self.index:
            return None

        return {ext: self._read_member(info) for ext, info in self.index[key].items()}

    def get_feature(self, key: str, feature_name: str) -> Optional[bytes]:
        """
        Get raw bytes for a single feature (extension) of a record.

        Seeks directly to the member's offset without reading other features.

        Args:
            key: The record key.
            feature_name: The feature name (file extension without dot, e.g. 'jpg').

        Returns:
            Raw bytes of the member, or None if key or feature is not found.
        """
        if key not in self.index:
            return None
        if feature_name not in self.index[key]:
            return None

        return self._read_member(self.index[key][feature_name])

    def get_feature_list(self, key: str, feature_name: str) -> Optional[List[bytes]]:
        """
        Get feature value as a single-element list.

        Consistent with TFRecordRandomAccess.get_feature_list(); for tar archives
        each feature appears exactly once per key.

        Returns:
            [bytes] if found, None otherwise.
        """
        value = self.get_feature(key, feature_name)
        if value is None:
            return None
        return [value]

    def contains_key(self, key: str) -> bool:
        """Check if a key exists in the index."""
        return key in self.index

    def get_keys(self) -> List[str]:
        """Return all indexed keys."""
        return list(self.index.keys())

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about the indexed records."""
        file_counts: Dict[str, int] = {}
        for key in self.index:
            file_path = next(iter(self.index[key].values()))['file']
            file_name = os.path.basename(file_path)
            file_counts[file_name] = file_counts.get(file_name, 0) + 1

        return {
            'total_records': len(self.index),
            'total_files': len(self.tar_files),
            'records_per_file': file_counts,
            'index_file': self.index_file,
        }

    def rebuild_index(self) -> None:
        """Force a full index rebuild, removing the cached file."""
        if os.path.exists(self.index_file):
            os.remove(self.index_file)
        self._index = None
        _ = self.index  # trigger rebuild immediately

    def close(self) -> None:
        """Close all open TarFile handles."""
        self.tarfile_pool.close_all()

    def __len__(self) -> int:
        return len(self.index)

    def __contains__(self, key: str) -> bool:
        return self.contains_key(key)

    def __getitem__(self, key: str) -> Dict[str, bytes]:
        result = self.get_record(key)
        if result is None:
            raise KeyError(f"Key '{key}' not found")
        return result

    def __enter__(self) -> 'TarRandomAccess':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
