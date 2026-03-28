"""
Unit tests for TarRandomAccess.
"""

import io
import json
import os
import shutil
import tarfile
import tempfile
import time

import pytest

from tfd_utils import TarRandomAccess
from tests.helpers.generate_test_data import create_test_tars


NUM_FILES = 3
RECORDS_PER_FILE = 5
TOTAL_RECORDS = NUM_FILES * RECORDS_PER_FILE


@pytest.fixture(scope="module")
def tar_test_data():
    """Creates 3 tar files × 5 records each in a temp directory."""
    test_dir = tempfile.mkdtemp(prefix="tfd_tar_test_")
    tar_files = create_test_tars(test_dir, num_files=NUM_FILES, records_per_file=RECORDS_PER_FILE)
    yield test_dir, tar_files
    shutil.rmtree(test_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInitialization:
    def test_single_file(self, tar_test_data):
        test_dir, tar_files = tar_test_data
        reader = TarRandomAccess(tar_files[0])
        assert len(reader.tar_files) == 1
        reader.close()

    def test_multiple_files_list(self, tar_test_data):
        test_dir, tar_files = tar_test_data
        reader = TarRandomAccess(tar_files)
        assert len(reader.tar_files) == NUM_FILES
        reader.close()

    def test_glob_pattern(self, tar_test_data):
        test_dir, tar_files = tar_test_data
        pattern = os.path.join(test_dir, "*.tar")
        reader = TarRandomAccess(pattern)
        assert len(reader.tar_files) == NUM_FILES
        reader.close()

    def test_invalid_path_raises(self):
        with pytest.raises(ValueError):
            TarRandomAccess("/nonexistent/path/to/nowhere.tar")

    def test_use_multiprocessing_disabled_for_single_file(self, tar_test_data):
        test_dir, tar_files = tar_test_data
        reader = TarRandomAccess(tar_files[0], use_multiprocessing=True)
        # Single file → multiprocessing should be disabled regardless
        assert reader.use_multiprocessing is False
        reader.close()


# ---------------------------------------------------------------------------
# Index structure and caching
# ---------------------------------------------------------------------------

class TestIndex:
    def test_index_has_correct_key_count(self, tar_test_data):
        test_dir, tar_files = tar_test_data
        reader = TarRandomAccess(tar_files)
        assert len(reader) == TOTAL_RECORDS
        reader.close()

    def test_index_file_created(self, tar_test_data):
        test_dir, tar_files = tar_test_data
        reader = TarRandomAccess(tar_files)
        _ = reader.index  # ensure index is built
        assert os.path.exists(reader.index_file)
        reader.close()

    def test_index_structure(self, tar_test_data):
        test_dir, tar_files = tar_test_data
        reader = TarRandomAccess(tar_files)
        for key, features in reader.index.items():
            assert isinstance(features, dict)
            for ext, info in features.items():
                assert 'file' in info
                assert 'member_name' in info
                assert 'data_offset' in info
                assert 'size' in info
                assert isinstance(info['data_offset'], int)
                assert isinstance(info['size'], int)
                assert info['size'] > 0
        reader.close()

    def test_index_file_naming_single(self, tar_test_data):
        test_dir, tar_files = tar_test_data
        reader = TarRandomAccess(tar_files[0])
        assert reader.index_file.endswith('.tar_index')
        reader.close()

    def test_index_file_naming_unified(self, tar_test_data):
        test_dir, tar_files = tar_test_data
        reader = TarRandomAccess(tar_files)
        assert '_unified.tar_index' in reader.index_file
        reader.close()

    def test_custom_index_file_path(self, tar_test_data):
        test_dir, tar_files = tar_test_data
        custom_path = os.path.join(test_dir, "custom.index")
        reader = TarRandomAccess(tar_files[0], index_file=custom_path)
        _ = reader.index
        assert reader.index_file == custom_path
        assert os.path.exists(custom_path)
        reader.close()


# ---------------------------------------------------------------------------
# get_record
# ---------------------------------------------------------------------------

class TestGetRecord:
    def test_get_record_returns_dict(self, tar_test_data):
        test_dir, tar_files = tar_test_data
        reader = TarRandomAccess(tar_files)
        key = reader.get_keys()[0]
        record = reader.get_record(key)
        assert isinstance(record, dict)
        reader.close()

    def test_get_record_has_expected_features(self, tar_test_data):
        test_dir, tar_files = tar_test_data
        reader = TarRandomAccess(tar_files)
        key = reader.get_keys()[0]
        record = reader.get_record(key)
        assert 'jpg' in record
        assert 'json' in record
        reader.close()

    def test_get_record_values_are_bytes(self, tar_test_data):
        test_dir, tar_files = tar_test_data
        reader = TarRandomAccess(tar_files)
        key = reader.get_keys()[0]
        record = reader.get_record(key)
        for val in record.values():
            assert isinstance(val, bytes)
        reader.close()

    def test_get_record_nonexistent_key_returns_none(self, tar_test_data):
        test_dir, tar_files = tar_test_data
        reader = TarRandomAccess(tar_files)
        assert reader.get_record("does_not_exist") is None
        reader.close()

    def test_get_record_content_correctness(self, tar_test_data):
        test_dir, tar_files = tar_test_data
        reader = TarRandomAccess(tar_files[0])
        # The first key in file 0 is test_000_0000
        key = "test_000_0000"
        record = reader.get_record(key)
        assert record is not None
        parsed = json.loads(record['json'].decode('utf-8'))
        assert parsed['key'] == key
        assert parsed['file_index'] == 0
        assert parsed['record_index'] == 0
        reader.close()


# ---------------------------------------------------------------------------
# get_feature
# ---------------------------------------------------------------------------

class TestGetFeature:
    def test_get_feature_returns_bytes(self, tar_test_data):
        test_dir, tar_files = tar_test_data
        reader = TarRandomAccess(tar_files)
        key = reader.get_keys()[0]
        value = reader.get_feature(key, 'jpg')
        assert isinstance(value, bytes)
        reader.close()

    def test_get_feature_missing_key_returns_none(self, tar_test_data):
        test_dir, tar_files = tar_test_data
        reader = TarRandomAccess(tar_files)
        assert reader.get_feature("no_such_key", 'jpg') is None
        reader.close()

    def test_get_feature_missing_ext_returns_none(self, tar_test_data):
        test_dir, tar_files = tar_test_data
        reader = TarRandomAccess(tar_files)
        key = reader.get_keys()[0]
        assert reader.get_feature(key, 'png') is None
        reader.close()

    def test_get_feature_independent_of_get_record(self, tar_test_data):
        """get_feature result should match the same feature from get_record."""
        test_dir, tar_files = tar_test_data
        reader = TarRandomAccess(tar_files)
        key = reader.get_keys()[0]
        via_feature = reader.get_feature(key, 'jpg')
        via_record = reader.get_record(key)['jpg']
        assert via_feature == via_record
        reader.close()

    def test_get_feature_json_parseable(self, tar_test_data):
        test_dir, tar_files = tar_test_data
        reader = TarRandomAccess(tar_files)
        key = "test_001_0002"
        value = reader.get_feature(key, 'json')
        assert value is not None
        parsed = json.loads(value.decode('utf-8'))
        assert parsed['key'] == key
        reader.close()


# ---------------------------------------------------------------------------
# get_feature_list
# ---------------------------------------------------------------------------

class TestGetFeatureList:
    def test_get_feature_list_returns_list(self, tar_test_data):
        test_dir, tar_files = tar_test_data
        reader = TarRandomAccess(tar_files)
        key = reader.get_keys()[0]
        result = reader.get_feature_list(key, 'jpg')
        assert isinstance(result, list)
        assert len(result) == 1
        reader.close()

    def test_get_feature_list_element_matches_get_feature(self, tar_test_data):
        test_dir, tar_files = tar_test_data
        reader = TarRandomAccess(tar_files)
        key = reader.get_keys()[0]
        assert reader.get_feature_list(key, 'jpg')[0] == reader.get_feature(key, 'jpg')
        reader.close()

    def test_get_feature_list_missing_returns_none(self, tar_test_data):
        test_dir, tar_files = tar_test_data
        reader = TarRandomAccess(tar_files)
        assert reader.get_feature_list("no_key", 'jpg') is None
        reader.close()


# ---------------------------------------------------------------------------
# Container protocol
# ---------------------------------------------------------------------------

class TestContainerProtocol:
    def test_len(self, tar_test_data):
        test_dir, tar_files = tar_test_data
        reader = TarRandomAccess(tar_files)
        assert len(reader) == TOTAL_RECORDS
        reader.close()

    def test_contains_existing_key(self, tar_test_data):
        test_dir, tar_files = tar_test_data
        reader = TarRandomAccess(tar_files)
        key = reader.get_keys()[0]
        assert key in reader
        reader.close()

    def test_contains_missing_key(self, tar_test_data):
        test_dir, tar_files = tar_test_data
        reader = TarRandomAccess(tar_files)
        assert "nonexistent" not in reader
        reader.close()

    def test_getitem_returns_dict(self, tar_test_data):
        test_dir, tar_files = tar_test_data
        reader = TarRandomAccess(tar_files)
        key = reader.get_keys()[0]
        assert isinstance(reader[key], dict)
        reader.close()

    def test_getitem_missing_raises_keyerror(self, tar_test_data):
        test_dir, tar_files = tar_test_data
        reader = TarRandomAccess(tar_files)
        with pytest.raises(KeyError):
            _ = reader["nonexistent_key"]
        reader.close()


# ---------------------------------------------------------------------------
# get_keys / contains_key
# ---------------------------------------------------------------------------

class TestKeys:
    def test_get_keys_returns_all(self, tar_test_data):
        test_dir, tar_files = tar_test_data
        reader = TarRandomAccess(tar_files)
        keys = reader.get_keys()
        assert len(keys) == TOTAL_RECORDS
        reader.close()

    def test_get_keys_are_strings(self, tar_test_data):
        test_dir, tar_files = tar_test_data
        reader = TarRandomAccess(tar_files)
        for k in reader.get_keys():
            assert isinstance(k, str)
        reader.close()

    def test_contains_key(self, tar_test_data):
        test_dir, tar_files = tar_test_data
        reader = TarRandomAccess(tar_files)
        assert reader.contains_key("test_000_0000")
        assert not reader.contains_key("nonexistent")
        reader.close()


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------

class TestGetStats:
    def test_stats_keys(self, tar_test_data):
        test_dir, tar_files = tar_test_data
        reader = TarRandomAccess(tar_files)
        stats = reader.get_stats()
        assert 'total_records' in stats
        assert 'total_files' in stats
        assert 'records_per_file' in stats
        assert 'index_file' in stats
        reader.close()

    def test_stats_total_records(self, tar_test_data):
        test_dir, tar_files = tar_test_data
        reader = TarRandomAccess(tar_files)
        assert reader.get_stats()['total_records'] == TOTAL_RECORDS
        reader.close()

    def test_stats_total_files(self, tar_test_data):
        test_dir, tar_files = tar_test_data
        reader = TarRandomAccess(tar_files)
        assert reader.get_stats()['total_files'] == NUM_FILES
        reader.close()

    def test_stats_records_per_file(self, tar_test_data):
        test_dir, tar_files = tar_test_data
        reader = TarRandomAccess(tar_files)
        rpf = reader.get_stats()['records_per_file']
        assert sum(rpf.values()) == TOTAL_RECORDS
        for count in rpf.values():
            assert count == RECORDS_PER_FILE
        reader.close()


# ---------------------------------------------------------------------------
# Index validity and rebuild
# ---------------------------------------------------------------------------

class TestIndexValidity:
    def test_index_valid_after_build(self, tar_test_data):
        test_dir, tar_files = tar_test_data
        reader = TarRandomAccess(tar_files[0])
        _ = reader.index
        assert reader._is_index_valid()
        reader.close()

    def test_index_invalid_when_missing(self, tar_test_data):
        test_dir, tar_files = tar_test_data
        tmp_dir = tempfile.mkdtemp()
        try:
            # Copy a tar file to a fresh dir with no index
            src = tar_files[0]
            dst = os.path.join(tmp_dir, os.path.basename(src))
            shutil.copy2(src, dst)
            reader = TarRandomAccess(dst)
            assert not reader._is_index_valid()
            reader.close()
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_index_invalid_after_tar_modified(self, tar_test_data):
        test_dir, tar_files = tar_test_data
        tmp_dir = tempfile.mkdtemp()
        try:
            src = tar_files[0]
            dst = os.path.join(tmp_dir, os.path.basename(src))
            shutil.copy2(src, dst)
            reader = TarRandomAccess(dst)
            _ = reader.index  # build index
            time.sleep(0.05)
            # Touch the tar file to make it newer than the index
            os.utime(dst, None)
            assert not reader._is_index_valid()
            reader.close()
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_rebuild_index(self, tar_test_data):
        test_dir, tar_files = tar_test_data
        tmp_dir = tempfile.mkdtemp()
        try:
            src = tar_files[0]
            dst = os.path.join(tmp_dir, os.path.basename(src))
            shutil.copy2(src, dst)
            reader = TarRandomAccess(dst)
            _ = reader.index
            index_path = reader.index_file
            assert os.path.exists(index_path)
            reader.rebuild_index()
            assert os.path.exists(index_path)
            assert len(reader) == RECORDS_PER_FILE
            reader.close()
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_index_valid_many_files_uses_parent_dir(self, tar_test_data):
        """With >5 files the parent directory mtime path is exercised."""
        test_dir, tar_files = tar_test_data
        tmp_dir = tempfile.mkdtemp()
        try:
            # Create 6 tar files in tmp_dir
            tar_paths = create_test_tars(tmp_dir, num_files=6, records_per_file=2)
            reader = TarRandomAccess(tar_paths)
            _ = reader.index  # build index
            assert reader._is_index_valid()
            reader.close()
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

class TestContextManager:
    def test_context_manager(self, tar_test_data):
        test_dir, tar_files = tar_test_data
        with TarRandomAccess(tar_files) as reader:
            assert len(reader) == TOTAL_RECORDS


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_tar(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            empty_tar = os.path.join(tmp_dir, "empty.tar")
            with tarfile.open(empty_tar, 'w'):
                pass
            reader = TarRandomAccess(empty_tar)
            assert len(reader) == 0
            assert reader.get_keys() == []
            assert reader.get_record("any") is None
            reader.close()
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_members_without_extension_are_skipped(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            tar_path = os.path.join(tmp_dir, "mixed.tar")
            with tarfile.open(tar_path, 'w') as tf:
                # Member with extension
                data = b"hello"
                info = tarfile.TarInfo(name="foo.txt")
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
                # Member without extension
                data2 = b"world"
                info2 = tarfile.TarInfo(name="README")
                info2.size = len(data2)
                tf.addfile(info2, io.BytesIO(data2))

            reader = TarRandomAccess(tar_path)
            assert len(reader) == 1
            assert "foo" in reader
            assert "README" not in reader
            reader.close()
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_subdirectory_keys(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            tar_path = os.path.join(tmp_dir, "subdir.tar")
            with tarfile.open(tar_path, 'w') as tf:
                for name in ("a/foo.jpg", "a/foo.json", "b/bar.jpg"):
                    data = name.encode()
                    info = tarfile.TarInfo(name=name)
                    info.size = len(data)
                    tf.addfile(info, io.BytesIO(data))

            reader = TarRandomAccess(tar_path)
            assert "a/foo" in reader
            assert "b/bar" in reader
            assert len(reader) == 2  # a/foo and b/bar
            record = reader["a/foo"]
            assert 'jpg' in record
            assert 'json' in record
            reader.close()
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_jpg_magic_bytes_preserved(self, tar_test_data):
        """JPEG-like bytes written by create_test_tars start with the SOI marker."""
        test_dir, tar_files = tar_test_data
        reader = TarRandomAccess(tar_files[0])
        jpg_bytes = reader.get_feature("test_000_0000", "jpg")
        assert jpg_bytes is not None
        assert jpg_bytes[:2] == b'\xff\xd8'
        reader.close()
