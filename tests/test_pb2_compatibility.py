"""
Test suite to verify that our pb2 protobuf classes are compatible with TensorFlow's protobuf classes.
"""

import os
import sys
import pytest
import tempfile
from typing import Any, Dict, List

# Import our module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tfd_utils.pb2 import (
    Example, SequenceExample, 
    Features, Feature, FeatureList, FeatureLists,
    BytesList, Int64List, FloatList
)

# Import TensorFlow
try:
    import tensorflow as tf
    from tensorflow.core.example import example_pb2 as tf_example_pb2
    from tensorflow.core.example import feature_pb2 as tf_feature_pb2
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available")
class TestPb2Compatibility:
    """Test compatibility between our protobuf classes and TensorFlow's protobuf classes."""
    
    def test_bytes_list_compatibility(self):
        """Test that BytesList is compatible with TensorFlow's BytesList."""
        test_values = [b"hello", b"world", b"test"]
        
        # Create using our implementation
        our_bytes_list = BytesList(value=test_values)
        
        # Create using TensorFlow's implementation
        tf_bytes_list = tf_feature_pb2.BytesList(value=test_values)
        
        # Test serialization compatibility
        our_serialized = our_bytes_list.SerializeToString()
        tf_serialized = tf_bytes_list.SerializeToString()
        
        assert our_serialized == tf_serialized, "BytesList serialization differs"
        
        # Test deserialization compatibility
        our_deserialized = BytesList()
        our_deserialized.ParseFromString(tf_serialized)
        assert list(our_deserialized.value) == test_values
        
        tf_deserialized = tf_feature_pb2.BytesList()
        tf_deserialized.ParseFromString(our_serialized)
        assert list(tf_deserialized.value) == test_values
    
    def test_int64_list_compatibility(self):
        """Test that Int64List is compatible with TensorFlow's Int64List."""
        test_values = [1, 2, 3, 100, -50]
        
        # Create using our implementation
        our_int64_list = Int64List(value=test_values)
        
        # Create using TensorFlow's implementation
        tf_int64_list = tf_feature_pb2.Int64List(value=test_values)
        
        # Test serialization compatibility
        our_serialized = our_int64_list.SerializeToString()
        tf_serialized = tf_int64_list.SerializeToString()
        
        assert our_serialized == tf_serialized, "Int64List serialization differs"
        
        # Test deserialization compatibility
        our_deserialized = Int64List()
        our_deserialized.ParseFromString(tf_serialized)
        assert list(our_deserialized.value) == test_values
        
        tf_deserialized = tf_feature_pb2.Int64List()
        tf_deserialized.ParseFromString(our_serialized)
        assert list(tf_deserialized.value) == test_values
    
    def test_float_list_compatibility(self):
        """Test that FloatList is compatible with TensorFlow's FloatList."""
        test_values = [1.0, 2.5, 3.14, -0.5]
        
        # Create using our implementation
        our_float_list = FloatList(value=test_values)
        
        # Create using TensorFlow's implementation
        tf_float_list = tf_feature_pb2.FloatList(value=test_values)
        
        # Test serialization compatibility
        our_serialized = our_float_list.SerializeToString()
        tf_serialized = tf_float_list.SerializeToString()
        
        assert our_serialized == tf_serialized, "FloatList serialization differs"
        
        # Test deserialization compatibility
        our_deserialized = FloatList()
        our_deserialized.ParseFromString(tf_serialized)
        
        # do not do this, float will lose precision
        # assert list(our_deserialized.value) == test_values
        
        tf_deserialized = tf_feature_pb2.FloatList()
        tf_deserialized.ParseFromString(our_serialized)
        # do not do this, float will lose precision
        # assert list(tf_deserialized.value) == test_values
        assert  list(our_deserialized.value) == list(tf_deserialized.value), "FloatList deserialization differs in values"
    
    def test_feature_compatibility(self):
        """Test that Feature is compatible with TensorFlow's Feature."""
        # Test with bytes_list
        bytes_feature_our = Feature(bytes_list=BytesList(value=[b"test"]))
        bytes_feature_tf = tf_feature_pb2.Feature(
            bytes_list=tf_feature_pb2.BytesList(value=[b"test"])
        )
        
        assert bytes_feature_our.SerializeToString() == bytes_feature_tf.SerializeToString()
        
        # Test with int64_list
        int64_feature_our = Feature(int64_list=Int64List(value=[42]))
        int64_feature_tf = tf_feature_pb2.Feature(
            int64_list=tf_feature_pb2.Int64List(value=[42])
        )
        
        assert int64_feature_our.SerializeToString() == int64_feature_tf.SerializeToString()
        
        # Test with float_list
        float_feature_our = Feature(float_list=FloatList(value=[3.14]))
        float_feature_tf = tf_feature_pb2.Feature(
            float_list=tf_feature_pb2.FloatList(value=[3.14])
        )
        
        assert float_feature_our.SerializeToString() == float_feature_tf.SerializeToString()
    
    def test_features_compatibility(self):
        """Test that Features is compatible with TensorFlow's Features."""
        feature_dict = {
            'bytes_feature': Feature(bytes_list=BytesList(value=[b"hello"])),
            'int64_feature': Feature(int64_list=Int64List(value=[42])),
            'float_feature': Feature(float_list=FloatList(value=[3.14])),
        }
        
        tf_feature_dict = {
            'bytes_feature': tf_feature_pb2.Feature(
                bytes_list=tf_feature_pb2.BytesList(value=[b"hello"])
            ),
            'int64_feature': tf_feature_pb2.Feature(
                int64_list=tf_feature_pb2.Int64List(value=[42])
            ),
            'float_feature': tf_feature_pb2.Feature(
                float_list=tf_feature_pb2.FloatList(value=[3.14])
            ),
        }
        
        # Create using our implementation
        our_features = Features(feature=feature_dict)
        
        # Create using TensorFlow's implementation
        tf_features = tf_feature_pb2.Features(feature=tf_feature_dict)
        
        # Test serialization compatibility
        our_serialized = our_features.SerializeToString()
        tf_serialized = tf_features.SerializeToString()
        
        assert our_serialized == tf_serialized, "Features serialization differs"
        
        # Test cross-deserialization
        our_deserialized = Features()
        our_deserialized.ParseFromString(tf_serialized)
        assert len(our_deserialized.feature) == 3
        
        tf_deserialized = tf_feature_pb2.Features()
        tf_deserialized.ParseFromString(our_serialized)
        assert len(tf_deserialized.feature) == 3
    
    def test_example_compatibility(self):
        """Test that Example is compatible with TensorFlow's Example."""
        features = {
            'text': Feature(bytes_list=BytesList(value=[b"hello world"])),
            'label': Feature(int64_list=Int64List(value=[1])),
            'score': Feature(float_list=FloatList(value=[0.95])),
        }
        
        tf_features = {
            'text': tf_feature_pb2.Feature(
                bytes_list=tf_feature_pb2.BytesList(value=[b"hello world"])
            ),
            'label': tf_feature_pb2.Feature(
                int64_list=tf_feature_pb2.Int64List(value=[1])
            ),
            'score': tf_feature_pb2.Feature(
                float_list=tf_feature_pb2.FloatList(value=[0.95])
            ),
        }
        
        # Create using our implementation
        our_example = Example(features=Features(feature=features))
        
        # Create using TensorFlow's implementation
        tf_example = tf_example_pb2.Example(
            features=tf_feature_pb2.Features(feature=tf_features)
        )
        
        # Test serialization compatibility
        our_serialized = our_example.SerializeToString()
        tf_serialized = tf_example.SerializeToString()
        
        assert our_serialized == tf_serialized, "Example serialization differs"
        
        # Test cross-deserialization
        our_deserialized = Example()
        our_deserialized.ParseFromString(tf_serialized)
        assert len(our_deserialized.features.feature) == 3
        
        tf_deserialized = tf_example_pb2.Example()
        tf_deserialized.ParseFromString(our_serialized)
        assert len(tf_deserialized.features.feature) == 3
    
    def test_feature_list_compatibility(self):
        """Test that FeatureList is compatible with TensorFlow's FeatureList."""
        features = [
            Feature(int64_list=Int64List(value=[1])),
            Feature(int64_list=Int64List(value=[2])),
            Feature(int64_list=Int64List(value=[3])),
        ]
        
        tf_features = [
            tf_feature_pb2.Feature(int64_list=tf_feature_pb2.Int64List(value=[1])),
            tf_feature_pb2.Feature(int64_list=tf_feature_pb2.Int64List(value=[2])),
            tf_feature_pb2.Feature(int64_list=tf_feature_pb2.Int64List(value=[3])),
        ]
        
        # Create using our implementation
        our_feature_list = FeatureList(feature=features)
        
        # Create using TensorFlow's implementation
        tf_feature_list = tf_feature_pb2.FeatureList(feature=tf_features)
        
        # Test serialization compatibility
        our_serialized = our_feature_list.SerializeToString()
        tf_serialized = tf_feature_list.SerializeToString()
        
        assert our_serialized == tf_serialized, "FeatureList serialization differs"
    
    def test_feature_lists_compatibility(self):
        """Test that FeatureLists is compatible with TensorFlow's FeatureLists."""
        feature_lists = {
            'sequence_feature': FeatureList(feature=[
                Feature(int64_list=Int64List(value=[1])),
                Feature(int64_list=Int64List(value=[2])),
            ])
        }
        
        tf_feature_lists = {
            'sequence_feature': tf_feature_pb2.FeatureList(feature=[
                tf_feature_pb2.Feature(int64_list=tf_feature_pb2.Int64List(value=[1])),
                tf_feature_pb2.Feature(int64_list=tf_feature_pb2.Int64List(value=[2])),
            ])
        }
        
        # Create using our implementation
        our_feature_lists = FeatureLists(feature_list=feature_lists)
        
        # Create using TensorFlow's implementation
        tf_feature_lists_obj = tf_feature_pb2.FeatureLists(feature_list=tf_feature_lists)
        
        # Test serialization compatibility
        our_serialized = our_feature_lists.SerializeToString()
        tf_serialized = tf_feature_lists_obj.SerializeToString()
        
        assert our_serialized == tf_serialized, "FeatureLists serialization differs"
    
    def test_sequence_example_compatibility(self):
        """Test that SequenceExample is compatible with TensorFlow's SequenceExample."""
        context = Features(feature={
            'global_feature': Feature(bytes_list=BytesList(value=[b"context"]))
        })
        
        feature_lists = FeatureLists(feature_list={
            'sequence_feature': FeatureList(feature=[
                Feature(int64_list=Int64List(value=[1])),
                Feature(int64_list=Int64List(value=[2])),
            ])
        })
        
        tf_context = tf_feature_pb2.Features(feature={
            'global_feature': tf_feature_pb2.Feature(
                bytes_list=tf_feature_pb2.BytesList(value=[b"context"])
            )
        })
        
        tf_feature_lists = tf_feature_pb2.FeatureLists(feature_list={
            'sequence_feature': tf_feature_pb2.FeatureList(feature=[
                tf_feature_pb2.Feature(int64_list=tf_feature_pb2.Int64List(value=[1])),
                tf_feature_pb2.Feature(int64_list=tf_feature_pb2.Int64List(value=[2])),
            ])
        })
        
        # Create using our implementation
        our_seq_example = SequenceExample(context=context, feature_lists=feature_lists)
        
        # Create using TensorFlow's implementation
        tf_seq_example = tf_example_pb2.SequenceExample(
            context=tf_context, 
            feature_lists=tf_feature_lists
        )
        
        # Test serialization compatibility
        our_serialized = our_seq_example.SerializeToString()
        tf_serialized = tf_seq_example.SerializeToString()
        
        assert our_serialized == tf_serialized, "SequenceExample serialization differs"
    
    def test_empty_values_compatibility(self):
        """Test that empty values are handled identically."""
        # Empty BytesList
        our_empty_bytes = BytesList(value=[])
        tf_empty_bytes = tf_feature_pb2.BytesList(value=[])
        assert our_empty_bytes.SerializeToString() == tf_empty_bytes.SerializeToString()
        
        # Empty Int64List
        our_empty_int64 = Int64List(value=[])
        tf_empty_int64 = tf_feature_pb2.Int64List(value=[])
        assert our_empty_int64.SerializeToString() == tf_empty_int64.SerializeToString()
        
        # Empty FloatList
        our_empty_float = FloatList(value=[])
        tf_empty_float = tf_feature_pb2.FloatList(value=[])
        assert our_empty_float.SerializeToString() == tf_empty_float.SerializeToString()
        
        # Empty Features
        our_empty_features = Features(feature={})
        tf_empty_features = tf_feature_pb2.Features(feature={})
        assert our_empty_features.SerializeToString() == tf_empty_features.SerializeToString()
    
    def test_large_values_compatibility(self):
        """Test that large values are handled identically."""
        # Large BytesList
        large_bytes = [b"x" * 1000 for _ in range(100)]
        our_large_bytes = BytesList(value=large_bytes)
        tf_large_bytes = tf_feature_pb2.BytesList(value=large_bytes)
        assert our_large_bytes.SerializeToString() == tf_large_bytes.SerializeToString()
        
        # Large Int64List
        large_ints = list(range(1000))
        our_large_ints = Int64List(value=large_ints)
        tf_large_ints = tf_feature_pb2.Int64List(value=large_ints)
        assert our_large_ints.SerializeToString() == tf_large_ints.SerializeToString()
        
        # Large FloatList
        large_floats = [float(i) * 0.1 for i in range(1000)]
        our_large_floats = FloatList(value=large_floats)
        tf_large_floats = tf_feature_pb2.FloatList(value=large_floats)
        assert our_large_floats.SerializeToString() == tf_large_floats.SerializeToString()
    
    def test_special_values_compatibility(self):
        """Test that special values are handled identically."""
        # Test with negative integers
        negative_values = [-1, -100, -9223372036854775808]  # Including min int64
        our_negative = Int64List(value=negative_values)
        tf_negative = tf_feature_pb2.Int64List(value=negative_values)
        assert our_negative.SerializeToString() == tf_negative.SerializeToString()
        
        # Test with special float values
        special_floats = [float('inf'), float('-inf'), 0.0, -0.0]
        our_special_floats = FloatList(value=special_floats)
        tf_special_floats = tf_feature_pb2.FloatList(value=special_floats)
        assert our_special_floats.SerializeToString() == tf_special_floats.SerializeToString()
        
        # Test with unicode bytes
        unicode_bytes = ["hello 世界".encode('utf-8'), "🎉".encode('utf-8')]
        our_unicode = BytesList(value=unicode_bytes)
        tf_unicode = tf_feature_pb2.BytesList(value=unicode_bytes)
        assert our_unicode.SerializeToString() == tf_unicode.SerializeToString()
    
    def test_tfrecord_roundtrip_compatibility(self):
        """Test that Examples can be written and read from TFRecord files interchangeably."""
        # Create test data using our implementation
        our_examples = []
        for i in range(5):
            features = {
                'id': Feature(int64_list=Int64List(value=[i])),
                'text': Feature(bytes_list=BytesList(value=[f"example_{i}".encode('utf-8')])),
                'score': Feature(float_list=FloatList(value=[float(i) * 0.1])),
            }
            our_examples.append(Example(features=Features(feature=features)))
        
        # Create equivalent data using TensorFlow's implementation
        tf_examples = []
        for i in range(5):
            features = {
                'id': tf_feature_pb2.Feature(
                    int64_list=tf_feature_pb2.Int64List(value=[i])
                ),
                'text': tf_feature_pb2.Feature(
                    bytes_list=tf_feature_pb2.BytesList(value=[f"example_{i}".encode('utf-8')])
                ),
                'score': tf_feature_pb2.Feature(
                    float_list=tf_feature_pb2.FloatList(value=[float(i) * 0.1])
                ),
            }
            tf_examples.append(tf_example_pb2.Example(
                features=tf_feature_pb2.Features(feature=features)
            ))
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write our examples to TFRecord
            our_file = os.path.join(temp_dir, "our_examples.tfrecord")
            with tf.io.TFRecordWriter(our_file) as writer:
                for example in our_examples:
                    writer.write(example.SerializeToString())
            
            # Write TensorFlow examples to TFRecord
            tf_file = os.path.join(temp_dir, "tf_examples.tfrecord")
            with tf.io.TFRecordWriter(tf_file) as writer:
                for example in tf_examples:
                    writer.write(example.SerializeToString())
            
            # Read both files and verify they can be parsed by both implementations
            for filename in [our_file, tf_file]:
                dataset = tf.data.TFRecordDataset(filename)
                for i, record in enumerate(dataset):
                    # Parse with our implementation
                    our_parsed = Example()
                    our_parsed.ParseFromString(record.numpy())
                    
                    # Parse with TensorFlow's implementation
                    tf_parsed = tf_example_pb2.Example()
                    tf_parsed.ParseFromString(record.numpy())
                    
                    # Verify both parse to the same content
                    assert our_parsed.SerializeToString() == tf_parsed.SerializeToString()
                    
                    # Verify specific field values
                    assert our_parsed.features.feature['id'].int64_list.value[0] == i
                    assert tf_parsed.features.feature['id'].int64_list.value[0] == i
                    assert our_parsed.features.feature['text'].bytes_list.value[0] == f"example_{i}".encode('utf-8')
                    assert tf_parsed.features.feature['text'].bytes_list.value[0] == f"example_{i}".encode('utf-8')
