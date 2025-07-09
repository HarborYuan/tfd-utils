"""
Test suite to verify that our TFRecordWriter is compatible with tf.io.TFRecordWriter.
"""

import os
import pytest
import tempfile
import shutil
from pathlib import Path
import hashlib
import filecmp

# Import our module
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tfd_utils.writer.tf_writer import TFRecordWriter
from tfd_utils.pb2 import Example, Features, Feature, BytesList, Int64List, FloatList

# Import TensorFlow
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def create_test_example(key: str, value: str, num_value: int) -> Example:
    """Create a test Example protobuf."""
    features = {
        'key': Feature(bytes_list=BytesList(value=[key.encode('utf-8')])),
        'value': Feature(bytes_list=BytesList(value=[value.encode('utf-8')])),
        'num_value': Feature(int64_list=Int64List(value=[num_value])),
    }
    return Example(features=Features(feature=features))


def create_test_data():
    """Create test data for writing to TFRecord files."""
    test_data = []
    for i in range(10):
        example = create_test_example(f"key_{i}", f"value_{i}", i)
        test_data.append(example.SerializeToString())
    return test_data


@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available")
class TestTFRecordWriterCompatibility:
    """Test compatibility between our TFRecordWriter and tf.io.TFRecordWriter."""
    
    def test_single_record_compatibility(self):
        """Test that single record written by both writers produces identical files."""
        test_data = create_test_data()
        record = test_data[0]  # Use first record
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write using our TFRecordWriter
            our_file = os.path.join(temp_dir, "our_writer.tfrecord")
            with TFRecordWriter(our_file) as writer:
                writer.write(record)
            
            # Write using tf.io.TFRecordWriter
            tf_file = os.path.join(temp_dir, "tf_writer.tfrecord")
            with tf.io.TFRecordWriter(tf_file) as writer:
                writer.write(record)
            
            # Compare file sizes
            our_size = os.path.getsize(our_file)
            tf_size = os.path.getsize(tf_file)
            assert our_size == tf_size, f"File sizes differ: ours={our_size}, tf={tf_size}"
            
            # Compare file contents byte by byte
            assert filecmp.cmp(our_file, tf_file, shallow=False), "Files are not identical"
    
    def test_multiple_records_compatibility(self):
        """Test that multiple records written by both writers produce identical files."""
        test_data = create_test_data()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write using our TFRecordWriter
            our_file = os.path.join(temp_dir, "our_writer.tfrecord")
            with TFRecordWriter(our_file) as writer:
                for record in test_data:
                    writer.write(record)
            
            # Write using tf.io.TFRecordWriter
            tf_file = os.path.join(temp_dir, "tf_writer.tfrecord")
            with tf.io.TFRecordWriter(tf_file) as writer:
                for record in test_data:
                    writer.write(record)
            
            # Compare file sizes
            our_size = os.path.getsize(our_file)
            tf_size = os.path.getsize(tf_file)
            assert our_size == tf_size, f"File sizes differ: ours={our_size}, tf={tf_size}"
            
            # Compare file contents byte by byte
            assert filecmp.cmp(our_file, tf_file, shallow=False), "Files are not identical"
    
    def test_empty_record_compatibility(self):
        """Test that empty records are handled identically."""
        empty_record = b""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write using our TFRecordWriter
            our_file = os.path.join(temp_dir, "our_writer.tfrecord")
            with TFRecordWriter(our_file) as writer:
                writer.write(empty_record)
            
            # Write using tf.io.TFRecordWriter
            tf_file = os.path.join(temp_dir, "tf_writer.tfrecord")
            with tf.io.TFRecordWriter(tf_file) as writer:
                writer.write(empty_record)
            
            # Compare file contents
            assert filecmp.cmp(our_file, tf_file, shallow=False), "Files are not identical"
    
    def test_large_record_compatibility(self):
        """Test that large records are handled identically."""
        # Create a large record (1MB of data)
        large_data = b"x" * (1024 * 1024)
        example = create_test_example("large_key", large_data.decode('utf-8'), 999999)
        large_record = example.SerializeToString()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write using our TFRecordWriter
            our_file = os.path.join(temp_dir, "our_writer.tfrecord")
            with TFRecordWriter(our_file) as writer:
                writer.write(large_record)
            
            # Write using tf.io.TFRecordWriter
            tf_file = os.path.join(temp_dir, "tf_writer.tfrecord")
            with tf.io.TFRecordWriter(tf_file) as writer:
                writer.write(large_record)
            
            # Compare file sizes
            our_size = os.path.getsize(our_file)
            tf_size = os.path.getsize(tf_file)
            assert our_size == tf_size, f"File sizes differ: ours={our_size}, tf={tf_size}"
            
            # Compare file contents
            assert filecmp.cmp(our_file, tf_file, shallow=False), "Files are not identical"
    
    def test_interleaved_writing_compatibility(self):
        """Test that files can be read by both TensorFlow and our implementation."""
        test_data = create_test_data()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write using our TFRecordWriter
            our_file = os.path.join(temp_dir, "our_writer.tfrecord")
            with TFRecordWriter(our_file) as writer:
                for record in test_data:
                    writer.write(record)
            
            # Read using TensorFlow's TFRecordDataset
            tf_dataset = tf.data.TFRecordDataset(our_file)
            tf_records = list(tf_dataset.as_numpy_iterator())
            
            # Verify that TensorFlow can read our file correctly
            assert len(tf_records) == len(test_data), "Number of records mismatch"
            
            for i, (tf_record, original_record) in enumerate(zip(tf_records, test_data)):
                assert tf_record == original_record, f"Record {i} differs"
    
    def test_binary_data_compatibility(self):
        """Test that binary data is handled identically."""
        # Create binary data with various byte values
        binary_data = bytes(range(256))
        
        # Create example with binary data
        features = {
            'binary_data': Feature(bytes_list=BytesList(value=[binary_data])),
            'size': Feature(int64_list=Int64List(value=[len(binary_data)])),
        }
        example = Example(features=Features(feature=features))
        record = example.SerializeToString()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write using our TFRecordWriter
            our_file = os.path.join(temp_dir, "our_writer.tfrecord")
            with TFRecordWriter(our_file) as writer:
                writer.write(record)
            
            # Write using tf.io.TFRecordWriter
            tf_file = os.path.join(temp_dir, "tf_writer.tfrecord")
            with tf.io.TFRecordWriter(tf_file) as writer:
                writer.write(record)
            
            # Compare file contents
            assert filecmp.cmp(our_file, tf_file, shallow=False), "Files are not identical"
    
    def test_hash_comparison(self):
        """Test that files have identical hash values."""
        test_data = create_test_data()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write using our TFRecordWriter
            our_file = os.path.join(temp_dir, "our_writer.tfrecord")
            with TFRecordWriter(our_file) as writer:
                for record in test_data:
                    writer.write(record)
            
            # Write using tf.io.TFRecordWriter
            tf_file = os.path.join(temp_dir, "tf_writer.tfrecord")
            with tf.io.TFRecordWriter(tf_file) as writer:
                for record in test_data:
                    writer.write(record)
            
            # Calculate SHA256 hashes
            def file_hash(filepath):
                hash_sha256 = hashlib.sha256()
                with open(filepath, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hash_sha256.update(chunk)
                return hash_sha256.hexdigest()
            
            our_hash = file_hash(our_file)
            tf_hash = file_hash(tf_file)
            
            assert our_hash == tf_hash, f"File hashes differ: ours={our_hash}, tf={tf_hash}"
