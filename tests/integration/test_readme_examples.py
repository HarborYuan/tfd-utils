import os
import shutil
import pytest
from tfd_utils.writer import TFRecordWriter
from tfd_utils.random_access import TFRecordRandomAccess
from tfd_utils.pb2 import Example, Features, Feature, BytesList

@pytest.fixture(scope="module")
def create_test_data():
    """Creates dummy tfrecord files for testing."""
    if not os.path.exists("test_data"):
        os.makedirs("test_data")

    # Create data.tfrecord
    with TFRecordWriter("test_data/data.tfrecord") as writer:
        for i in range(10):
            key = f'record_{i}'.encode('utf-8')
            example = Example(features=Features(feature={
                'key': Feature(bytes_list=BytesList(value=[key])),
                'image': Feature(bytes_list=BytesList(value=[b'\xDE\xAD\xBE\xEF'])),
                'label': Feature(bytes_list=BytesList(value=[b'cat']))
            }))
            writer.write(example.SerializeToString())

    # Create custom_key.tfrecord
    with TFRecordWriter("test_data/custom_key.tfrecord") as writer:
        for i in range(5):
            example = Example(features=Features(feature={
                'user_id': Feature(bytes_list=BytesList(value=[f'user_{i}'.encode('utf-8')])),
                'data': Feature(bytes_list=BytesList(value=[b'some_data']))
            }))
            writer.write(example.SerializeToString())

    yield

    # Teardown
    shutil.rmtree("test_data")

def test_01_write_tfrecords(create_test_data):
    """Tests the writing of tfrecords."""
    assert os.path.exists("test_data/data.tfrecord")

def test_02_random_access_reading(create_test_data):
    """Tests random access reading."""
    reader = TFRecordRandomAccess("test_data/data.tfrecord")
    assert len(reader) == 10
    assert "record_3" in reader
    example = reader["record_3"]
    assert isinstance(example, Example)
    assert example.features.feature['image'].bytes_list.value[0] == b'\xDE\xAD\xBE\xEF'

def test_03_tensorflow_interoperability(create_test_data):
    """Tests reading with TensorFlow."""
    try:
        import tensorflow as tf
    except ImportError:
        pytest.skip("TensorFlow is not installed, skipping test")

    file_path = "test_data/data.tfrecord"
    dataset = tf.data.TFRecordDataset(file_path)

    def parse_tfrecord_fn(record):
        feature_description = {
            'key': tf.io.FixedLenFeature([], tf.string),
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.string),
        }
        return tf.io.parse_single_example(record, feature_description)

    parsed_dataset = dataset.map(parse_tfrecord_fn)
    for i, parsed_record in enumerate(parsed_dataset):
        assert parsed_record['key'].numpy() == f'record_{i}'.encode('utf-8')

def test_04_advanced_usage(create_test_data):
    """Tests advanced usage scenarios."""
    # Custom key feature
    reader_custom_key = TFRecordRandomAccess("test_data/custom_key.tfrecord", key_feature_name="user_id")
    assert "user_3" in reader_custom_key
    example = reader_custom_key["user_3"]
    assert isinstance(example, Example)
    assert example.features.feature['data'].bytes_list.value[0] == b'some_data'

    # Custom index caching
    reader_custom_index = TFRecordRandomAccess(
        "test_data/custom_key.tfrecord",
        key_feature_name="user_id",
        index_file="test_data/my_custom_index.cache"
    )
    _ = reader_custom_index["user_1"]
    assert os.path.exists("test_data/my_custom_index.cache")
    reader_custom_index.rebuild_index()
