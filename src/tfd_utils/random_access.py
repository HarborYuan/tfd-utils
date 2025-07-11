"""
TFRecord Random Access Reader

This module provides a class for efficient random access to TFRecord files.
It builds an index on first access and caches it for subsequent lookups.
"""

import os
import pickle
import glob
from typing import Dict, Any, Optional, List, Union, BinaryIO
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from functools import partial
from collections import OrderedDict

from .pb2 import Example


class FileContextPool:
    """
    A thread-safe LRU cache for file handles to optimize random access operations.
    
    This pool maintains a limited number of open file handles to avoid
    repeatedly opening and closing files during random access operations.
    """
    
    def __init__(self, max_size: int = 100):
        """
        Initialize the file context pool.
        
        Args:
            max_size: Maximum number of file handles to keep open
        """
        self.max_size = max_size
        self._pool: OrderedDict[str, BinaryIO] = OrderedDict()
    
    def get_file_handle(self, file_path: str) -> BinaryIO:
        """
        Get a file handle from the pool, opening if necessary.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File handle ready for reading
        """
        if file_path in self._pool:
            # Move to end (most recently used)
            handle = self._pool.pop(file_path)
            self._pool[file_path] = handle
            return handle
        
        # Open new file
        handle = open(file_path, 'rb')
        
        # Check if we need to evict the least recently used file
        if len(self._pool) >= self.max_size:
            # Remove the least recently used (first) item
            oldest_path, oldest_handle = self._pool.popitem(last=False)
            try:
                oldest_handle.close()
            except Exception:
                pass  # Ignore errors when closing
        
        self._pool[file_path] = handle
        return handle
    
    def close_all(self):
        """Close all file handles in the pool."""
        for handle in self._pool.values():
            try:
                handle.close()
            except Exception:
                pass  # Ignore errors when closing
        self._pool.clear()
    
    def __del__(self):
        """Cleanup when the pool is destroyed."""
        self.close_all()


class TFRecordRandomAccess:
    """
    A class for random access to TFRecord files with automatic index caching.
    
    This class provides efficient random access to TFRecord files by building
    an index that maps keys to file positions. The index is built on first
    access and cached for subsequent uses.
    """
    
    def __init__(self, 
                 tfrecord_path: Union[str, Path, List[str], List[Path]], 
                 key_feature_name: str = 'key',
                 index_file: Optional[Union[str, Path]] = None,
                 progress_interval: int = 1024,
                 max_workers: Optional[int] = None,
                 use_multiprocessing: bool = True,
                 file_pool_size: int = 100):
        """
        Initialize the TFRecord random access reader.
        
        Args:
            tfrecord_path: Path to TFRecord file(s). Can be:
                - Single file path (str or Path)
                - List of file paths
                - Glob pattern (str) for multiple files
            key_feature_name: Name of the feature containing the record key
            index_file: Optional path to save/load the index cache. If None,
                       will be auto-generated based on tfrecord_path
            progress_interval: Print progress every N records during indexing
            max_workers: Maximum number of worker processes. If None, uses CPU count
            use_multiprocessing: Whether to use multiprocessing for parallel file processing
            file_pool_size: Maximum number of file handles to keep in the pool
        """
        self.key_feature_name = key_feature_name
        self.progress_interval = progress_interval
        self.max_workers = max_workers or multiprocessing.cpu_count()
        
        # Initialize file context pool
        self.file_pool = FileContextPool(max_size=file_pool_size)
        
        # Resolve TFRecord files
        self.tfrecord_files = self._resolve_tfrecord_files(tfrecord_path)
        if not self.tfrecord_files:
            raise ValueError(f"No TFRecord files found for path: {tfrecord_path}")
        
        self.use_multiprocessing = use_multiprocessing and len(self.tfrecord_files) > 1
        
        # Set up index file path
        self.index_file = self._get_index_file_path(index_file)
        
        # Initialize index
        self._index: Optional[Dict[str, Dict[str, Any]]] = None
    
    def _resolve_tfrecord_files(self, tfrecord_path: Union[str, Path, List[str], List[Path]]) -> List[str]:
        """Resolve the input path(s) to a list of TFRecord file paths."""
        if isinstance(tfrecord_path, (list, tuple)):
            # List of paths
            files = []
            for path in tfrecord_path:
                path_str = str(path)
                if os.path.exists(path_str):
                    files.append(path_str)
                else:
                    # Try as glob pattern
                    files.extend(glob.glob(path_str))
            return sorted(files)
        else:
            # Single path (string or Path)
            path_str = str(tfrecord_path)
            if os.path.exists(path_str):
                return [path_str]
            else:
                # Try as glob pattern
                return sorted(glob.glob(path_str))
    
    def _get_index_file_path(self, index_file: Optional[Union[str, Path]]) -> str:
        """Generate index file path if not provided."""
        if index_file is not None:
            return str(index_file)
        
        # Generate based on first TFRecord file
        first_file = Path(self.tfrecord_files[0])
        if len(self.tfrecord_files) == 1:
            # Single file: use same directory with .index extension
            return str(first_file.with_suffix('.index'))
        else:
            return str(first_file.parent / f"{first_file.stem}_unified.index")
    
    def _is_index_valid(self) -> bool:
        """Check if the cached index is still valid."""
        if not os.path.exists(self.index_file):
            return False
        
        index_mtime = os.path.getmtime(self.index_file)
        
        # For performance optimization: if there are many files, only check the first 5
        # and also check if the parent directory has been modified
        if len(self.tfrecord_files) > 5:
            # Check parent directory modification time
            # Note: Allow small time difference to avoid conflicts when index file creation
            # modifies the parent directory timestamp
            parent_dir = os.path.dirname(self.tfrecord_files[0])
            parent_mtime = os.path.getmtime(parent_dir)
            
            # If parent directory is significantly newer than index (more than 1 second),
            # it suggests files may have been added/removed
            if parent_mtime > index_mtime + 0.5:
                return False
            
            # Check only the first 5 files
            files_to_check = self.tfrecord_files[:5]
        else:
            # Check all files if there are 5 or fewer
            files_to_check = self.tfrecord_files
        
        # Check if any TFRecord file is newer than the index
        for tfrecord_file in files_to_check:
            if not os.path.exists(tfrecord_file):
                return False
            if os.path.getmtime(tfrecord_file) > index_mtime:
                return False

        return True
    
    def _build_index(self) -> Dict[str, Dict[str, Any]]:
        """Build index for all TFRecord files."""
        print(f"Building index for {len(self.tfrecord_files)} TFRecord file(s)...")
        
        if self.use_multiprocessing and len(self.tfrecord_files) > 1:
            return self._build_index_parallel()
        else:
            return self._build_index_sequential()
    
    def _build_index_sequential(self) -> Dict[str, Dict[str, Any]]:
        """Build index sequentially (original method)."""
        index = {}
        total_records = 0
        
        for tfrecord_file in self.tfrecord_files:
            file_index = _process_single_tfrecord(tfrecord_file, self.key_feature_name, self.progress_interval)
            index.update(file_index)
            total_records += len(file_index)
        
        print(f"Total records indexed: {total_records}")
        
        # Save the index to cache file
        with open(self.index_file, 'wb') as f:
            pickle.dump(index, f)
        print(f"Index saved to {self.index_file}")
        
        return index
    
    def _build_index_parallel(self) -> Dict[str, Dict[str, Any]]:
        """Build index using multiprocessing for parallel file processing."""
        print(f"Using {self.max_workers} worker processes for parallel indexing...")
        
        index = {}
        total_records = 0
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all files for processing - use optimized function
            process_func = partial(_process_single_tfrecord, 
                                 key_feature_name=self.key_feature_name,
                                 progress_interval=self.progress_interval)
            
            future_to_file = {executor.submit(process_func, tfrecord_file): tfrecord_file 
                             for tfrecord_file in self.tfrecord_files}
            
            # Collect results as they complete
            for future in as_completed(future_to_file):
                tfrecord_file = future_to_file[future]
                try:
                    file_index = future.result()
                    index.update(file_index)
                    total_records += len(file_index)
                except Exception as e:
                    print(f"Error processing {tfrecord_file}: {e}")
        
        print(f"Total records indexed: {total_records}")
        
        # Save the index to cache file
        with open(self.index_file, 'wb') as f:
            pickle.dump(index, f)
        print(f"Index saved to {self.index_file}")
        
        return index
    
    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load index from cache file or build if not exists."""
        if self._is_index_valid():
            print(f"Loading index from {self.index_file}")
            with open(self.index_file, 'rb') as f:
                return pickle.load(f)
        else:
            print("Index cache is invalid or missing, rebuilding...")
            return self._build_index()
    
    @property
    def index(self) -> Dict[str, Dict[str, Any]]:
        """Get the index, building it if necessary."""
        if self._index is None:
            self._index = self._load_index()
        return self._index
    
    def get_record(self, key: str) -> Optional[Example]:
        """
        Get a TFRecord by key.
        
        Args:
            key: The key to lookup
            
        Returns:
            Example if found, None otherwise
        """
        if key not in self.index:
            return None
        
        record_info = self.index[key]
        tfrecord_file = record_info['file']
        offset = record_info['offset']
        
        f = self.file_pool.get_file_handle(tfrecord_file)
        
        f.seek(offset)
            
        # Read the record at the given offset
        len_bytes = f.read(8)
        length = int.from_bytes(len_bytes, 'little')
            
        # Skip length CRC
        f.seek(4, os.SEEK_CUR)
            
        # Read record data
        record_bytes = f.read(length)
        
        # Parse and return the example
        return Example.FromString(record_bytes)
    
    def get_feature(self, key: str, feature_name: str) -> Optional[Any]:
        """
        Get a specific feature value from a record.
        
        Args:
            key: The key to lookup
            feature_name: Name of the feature to extract
            
        Returns:
            Feature value if found, None otherwise
        """
        example = self.get_record(key)
        if example is None:
            return None
        
        if feature_name not in example.features.feature:
            return None
        
        feature = example.features.feature[feature_name]
        
        # Return the appropriate value based on feature type
        if feature.bytes_list.value:
            return feature.bytes_list.value[0]
        elif feature.int64_list.value:
            return feature.int64_list.value[0]
        elif feature.float_list.value:
            return feature.float_list.value[0]
        else:
            return None
    
    def contains_key(self, key: str) -> bool:
        """Check if a key exists in the index."""
        return key in self.index
    
    def get_keys(self) -> List[str]:
        """Get all available keys."""
        return list(self.index.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the indexed records."""
        file_counts = {}
        for key, info in self.index.items():
            file_path = info['file']
            file_name = os.path.basename(file_path)
            file_counts[file_name] = file_counts.get(file_name, 0) + 1
        
        return {
            'total_records': len(self.index),
            'total_files': len(self.tfrecord_files),
            'records_per_file': file_counts,
            'index_file': self.index_file
        }
    
    def rebuild_index(self) -> None:
        """Force rebuild the index."""
        if os.path.exists(self.index_file):
            os.remove(self.index_file)
        self._index = None
        # Trigger rebuild on next access
        _ = self.index
    
    def __len__(self) -> int:
        """Return the number of records in the index."""
        return len(self.index)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists using 'in' operator."""
        return self.contains_key(key)
    
    def __getitem__(self, key: str) -> Example:
        """Get record using [] operator."""
        result = self.get_record(key)
        if result is None:
            raise KeyError(f"Key '{key}' not found")
        return result
    
    def close(self):
        """Close all file handles in the pool."""
        self.file_pool.close_all()
    
    def __del__(self):
        """Cleanup when the instance is destroyed."""
        try:
            self.close()
        except Exception:
            pass  # Ignore errors during cleanup
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

def _process_single_tfrecord(tfrecord_file: str, key_feature_name: str, progress_interval: int = 1000) -> Dict[str, Dict[str, Any]]:
    """Process a single TFRecord file and return its index."""
    print(f"Processing {os.path.basename(tfrecord_file)}...")
    
    index = {}
    file_records = 0
    
    with open(tfrecord_file, 'rb') as f:
        while True:
            offset = f.tell()
            try:
                # Read TFRecord format: [length][length_crc][data][data_crc]
                len_bytes = f.read(8)
                if not len_bytes:
                    break
                
                length = int.from_bytes(len_bytes, 'little')
                
                # Skip the CRC checksum for the length
                f.seek(4, os.SEEK_CUR)
                
                # Read the record data
                record_bytes = f.read(length)
                if len(record_bytes) != length:
                    break
                
                # Skip the CRC checksum for the record
                f.seek(4, os.SEEK_CUR)
                
                # Parse the record to extract the key
                example = Example.FromString(record_bytes)
                
                # Extract key from the specified feature
                if key_feature_name not in example.features.feature:
                    raise ValueError(f"Feature '{key_feature_name}' not found in record")
                
                feature = example.features.feature[key_feature_name]
                if feature.bytes_list.value:
                    key = feature.bytes_list.value[0].decode('utf-8')
                elif feature.int64_list.value:
                    key = str(feature.int64_list.value[0])
                elif feature.float_list.value:
                    key = str(feature.float_list.value[0])
                else:
                    raise ValueError(f"Unsupported feature type for key: {key_feature_name}")
                
                # Store file path and offset in the index
                index[key] = {
                    'file': tfrecord_file,
                    'offset': offset,
                    'length': length
                }
                
                file_records += 1
                
                if file_records % progress_interval == 0:
                    print(f"  Processed {file_records} records from {os.path.basename(tfrecord_file)}")
            
            except Exception as e:
                print(f"Error reading record at offset {offset} in {tfrecord_file}: {e}")
                break
    
    print(f"  Completed {os.path.basename(tfrecord_file)}: {file_records} records")
    return index
