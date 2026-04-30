"""
Test suite for TFRecord Random Access functionality.
"""

import os
import pytest
import tempfile
import shutil
import time
from pathlib import Path
import json

# Import our module
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from tfd_utils.random_access import TFRecordRandomAccess, _detect_complete_shard_set
from tests.helpers.generate_test_data import create_test_tfrecords, create_test_data_with_different_key_types


class TestTFRecordRandomAccess:
    """Test cases for TFRecordRandomAccess class."""
    
    @pytest.fixture(scope="class")
    def test_data_dir(self):
        """Create test data directory and files."""
        test_dir = tempfile.mkdtemp(prefix="tfd_utils_test_")
        
        # Create test TFRecord files
        tfrecord_files = create_test_tfrecords(test_dir, num_files=3, records_per_file=5)
        different_keys_file = create_test_data_with_different_key_types(test_dir)
        
        yield test_dir, tfrecord_files, different_keys_file
        
        # Cleanup
        shutil.rmtree(test_dir)
    
    def test_single_file_initialization(self, test_data_dir):
        """Test initialization with a single TFRecord file."""
        test_dir, tfrecord_files, _ = test_data_dir
        
        # Test with single file
        reader = TFRecordRandomAccess(tfrecord_files[0])
        assert len(reader.tfrecord_files) == 1
        assert reader.tfrecord_files[0] == tfrecord_files[0]
    
    def test_multiple_files_initialization(self, test_data_dir):
        """Test initialization with multiple TFRecord files."""
        test_dir, tfrecord_files, _ = test_data_dir
        
        # Test with list of files
        reader = TFRecordRandomAccess(tfrecord_files)
        assert len(reader.tfrecord_files) == len(tfrecord_files)
        assert set(reader.tfrecord_files) == set(tfrecord_files)
    
    def test_glob_pattern_initialization(self, test_data_dir):
        """Test initialization with glob pattern."""
        test_dir, tfrecord_files, _ = test_data_dir
        
        # Test with glob pattern
        pattern = os.path.join(test_dir, "test_data_*.tfrecord")
        reader = TFRecordRandomAccess(pattern)
        assert len(reader.tfrecord_files) == len(tfrecord_files)
    
    def test_invalid_path_initialization(self):
        """Test initialization with invalid path."""
        with pytest.raises(ValueError, match="No TFRecord files found"):
            TFRecordRandomAccess("/nonexistent/path/to/file.tfrecord")
    
    def test_index_building(self, test_data_dir):
        """Test index building functionality."""
        test_dir, tfrecord_files, _ = test_data_dir
        
        reader = TFRecordRandomAccess(tfrecord_files)
        
        # Index should be built on first access
        index = reader.index
        assert isinstance(index, dict)
        assert len(index) == 15  # 3 files × 5 records each
        
        # Check that index file was created
        assert os.path.exists(reader.index_file)
    
    def test_get_record(self, test_data_dir):
        """Test getting records by key."""
        test_dir, tfrecord_files, _ = test_data_dir
        
        reader = TFRecordRandomAccess(tfrecord_files)
        
        # Test getting existing record
        key = "test_000_0000"
        record = reader.get_record(key)
        assert record is not None
        
        # Verify the record content
        key_feature = record.features.feature['key'].bytes_list.value[0].decode('utf-8')
        assert key_feature == key
        
        # Test getting non-existent record
        non_existent = reader.get_record("non_existent_key")
        assert non_existent is None
    
    def test_get_feature(self, test_data_dir):
        """Test getting specific features from records."""
        test_dir, tfrecord_files, _ = test_data_dir
        
        reader = TFRecordRandomAccess(tfrecord_files)
        
        key = "test_000_0000"
        
        # Test getting image feature
        image_bytes = reader.get_feature(key, 'image')
        assert image_bytes is not None
        assert isinstance(image_bytes, bytes)
        
        # Test getting metadata feature
        metadata_bytes = reader.get_feature(key, 'metadata')
        assert metadata_bytes is not None
        metadata = json.loads(metadata_bytes.decode('utf-8'))
        assert isinstance(metadata, dict)
        assert 'width' in metadata
        
        # Test getting non-existent feature
        non_existent = reader.get_feature(key, 'non_existent_feature')
        assert non_existent is None
    
    def test_contains_key(self, test_data_dir):
        """Test key existence checking."""
        test_dir, tfrecord_files, _ = test_data_dir
        
        reader = TFRecordRandomAccess(tfrecord_files)
        
        # Test existing key
        assert reader.contains_key("test_000_0000")
        assert "test_000_0000" in reader  # Test __contains__ method
        
        # Test non-existent key
        assert not reader.contains_key("non_existent_key")
        assert "non_existent_key" not in reader
    
    def test_get_keys(self, test_data_dir):
        """Test getting all keys."""
        test_dir, tfrecord_files, _ = test_data_dir
        
        reader = TFRecordRandomAccess(tfrecord_files)
        
        keys = reader.get_keys()
        assert len(keys) == 15  # 3 files × 5 records each
        assert "test_000_0000" in keys
        assert "test_002_0004" in keys
    
    def test_get_stats(self, test_data_dir):
        """Test getting statistics."""
        test_dir, tfrecord_files, _ = test_data_dir
        
        reader = TFRecordRandomAccess(tfrecord_files)
        
        stats = reader.get_stats()
        assert stats['total_records'] == 15
        assert stats['total_files'] == 3
        assert 'records_per_file' in stats
        assert 'index_file' in stats
    
    def test_len_and_getitem(self, test_data_dir):
        """Test __len__ and __getitem__ methods."""
        test_dir, tfrecord_files, _ = test_data_dir
        
        reader = TFRecordRandomAccess(tfrecord_files)
        
        # Test __len__
        assert len(reader) == 15
        
        # Test __getitem__
        key = "test_000_0000"
        record = reader[key]
        assert record is not None
        
        # Test __getitem__ with non-existent key
        with pytest.raises(KeyError):
            _ = reader["non_existent_key"]
    
    def test_different_key_types(self, test_data_dir):
        """Test handling different key feature types."""
        test_dir, _, different_keys_file = test_data_dir
        
        # Test with string key
        reader1 = TFRecordRandomAccess(different_keys_file, key_feature_name='key')
        assert reader1.contains_key('string_key_001')
        
        # Test with integer key
        reader2 = TFRecordRandomAccess(different_keys_file, key_feature_name='id')
        reader2.rebuild_index()  # Ensure index is built
        print(reader2.index)  # Debugging output
        assert reader2.contains_key('1')
        
        # Test with float key
        reader3 = TFRecordRandomAccess(different_keys_file, key_feature_name='score')
        reader3.rebuild_index()  # Ensure index is built
        assert reader3.contains_key('99.5')
    
    def test_rebuild_index(self, test_data_dir):
        """Test index rebuilding functionality."""
        test_dir, tfrecord_files, _ = test_data_dir
        
        reader = TFRecordRandomAccess(tfrecord_files)
        
        # Build index first
        _ = reader.index
        index_file = reader.index_file
        assert os.path.exists(index_file)
        
        # Rebuild index
        reader.rebuild_index()
        
        # Index should still exist and work
        assert os.path.exists(index_file)
        assert len(reader.index) == 15
    
    def test_custom_index_file_path(self, test_data_dir):
        """Test custom index file path."""
        test_dir, tfrecord_files, _ = test_data_dir
        
        custom_index_path = os.path.join(test_dir, "custom_index.idx")
        reader = TFRecordRandomAccess(tfrecord_files, index_file=custom_index_path)
        
        # Build index
        _ = reader.index
        
        # Check custom index file was created
        assert os.path.exists(custom_index_path)
        assert reader.index_file == custom_index_path
    
    def test_progress_interval(self, test_data_dir):
        """Test progress interval setting."""
        test_dir, tfrecord_files, _ = test_data_dir
        
        # Test with different progress interval
        reader = TFRecordRandomAccess(tfrecord_files, progress_interval=2)
        
        # Should work without issues
        _ = reader.index
        assert len(reader.index) == 15
    
    def test_is_index_valid(self, test_data_dir):
        """_is_index_valid is now a pure existence check on the index file path."""
        test_dir, tfrecord_files, _ = test_data_dir

        reader = TFRecordRandomAccess(tfrecord_files)

        # Missing → invalid
        if os.path.exists(reader.index_file):
            os.remove(reader.index_file)
        assert not reader._is_index_valid()

        # Built → valid
        _ = reader.index
        assert reader._is_index_valid()

        # Touching source files no longer invalidates the index — mtime is
        # intentionally not checked under the new naming scheme.
        time.sleep(0.05)
        os.utime(tfrecord_files[0])
        assert reader._is_index_valid()

    def test_index_file_naming_single(self, test_data_dir):
        """Single-file reader: index path is <stem>.index."""
        _, tfrecord_files, _ = test_data_dir
        reader = TFRecordRandomAccess(tfrecord_files[0])
        expected = str(Path(tfrecord_files[0]).with_suffix('.index'))
        assert reader.index_file == expected

    def test_index_file_naming_unified_tot(self, test_data_dir):
        """Multi-file reader without shard pattern: <first_stem>_unified_tot<N>.index."""
        test_dir, tfrecord_files, _ = test_data_dir
        reader = TFRecordRandomAccess(tfrecord_files)
        first_stem = Path(tfrecord_files[0]).stem
        n = len(tfrecord_files)
        assert reader.index_file == os.path.join(test_dir, f"{first_stem}_unified_tot{n}.index")

    def test_index_file_naming_all_shards(self, tmp_path):
        """Complete shard set XXXXX_of_NNNNN.tfrecord → all<NNNNN+1>.index."""
        # Create 4 empty shards: 00000_of_00003 .. 00003_of_00003 (total = 4)
        shards = []
        for i in range(4):
            p = tmp_path / f"{i:05d}_of_{3:05d}.tfrecord"
            p.write_bytes(b"")
            shards.append(str(p))

        reader = TFRecordRandomAccess(shards)
        assert reader.index_file == str(tmp_path / "all4.index")

    def test_index_file_naming_incomplete_shards_falls_back(self, tmp_path):
        """Missing a shard → not a complete set → fall back to _unified_tot<N>.index."""
        # 3 of 4 shards present
        shards = []
        for i in [0, 1, 3]:
            p = tmp_path / f"{i:05d}_of_{3:05d}.tfrecord"
            p.write_bytes(b"")
            shards.append(str(p))

        reader = TFRecordRandomAccess(shards)
        first_stem = Path(shards[0]).stem
        assert reader.index_file == str(tmp_path / f"{first_stem}_unified_tot3.index")

    def test_index_path_changes_when_file_count_changes(self, tmp_path):
        """File count is encoded in the path: adding/removing files routes to a fresh path."""
        f1 = tmp_path / "a.tfrecord"; f1.write_bytes(b"")
        f2 = tmp_path / "b.tfrecord"; f2.write_bytes(b"")
        f3 = tmp_path / "c.tfrecord"; f3.write_bytes(b"")

        r2 = TFRecordRandomAccess([str(f1), str(f2)])
        r3 = TFRecordRandomAccess([str(f1), str(f2), str(f3)])
        assert r2.index_file != r3.index_file
        assert "_tot2.index" in r2.index_file
        assert "_tot3.index" in r3.index_file


class TestDetectCompleteShardSet:
    """Direct unit tests for the shard-pattern detector."""

    def test_complete_set(self, tmp_path):
        files = [str(tmp_path / f"{i:05d}_of_{2:05d}.tfrecord") for i in range(3)]
        assert _detect_complete_shard_set(files) == 3

    def test_complete_set_user_example(self, tmp_path):
        # 00000_of_09999 → all 10000 shards present → returns 10000
        files = [str(tmp_path / f"{i:05d}_of_09999.tfrecord") for i in range(10000)]
        assert _detect_complete_shard_set(files) == 10000

    def test_missing_shard(self, tmp_path):
        files = [str(tmp_path / f"{i:05d}_of_{2:05d}.tfrecord") for i in (0, 2)]
        assert _detect_complete_shard_set(files) is None

    def test_inconsistent_total(self, tmp_path):
        files = [
            str(tmp_path / "00000_of_00002.tfrecord"),
            str(tmp_path / "00001_of_00003.tfrecord"),
        ]
        assert _detect_complete_shard_set(files) is None

    def test_non_shard_filenames(self, tmp_path):
        files = [str(tmp_path / "test_data_000.tfrecord")]
        assert _detect_complete_shard_set(files) is None

    def test_handles_compound_extension(self, tmp_path):
        # The pattern only requires a leading dot, so compound extensions still match.
        files = [str(tmp_path / f"{i:05d}_of_{1:05d}.tar.gz") for i in range(2)]
        assert _detect_complete_shard_set(files) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
