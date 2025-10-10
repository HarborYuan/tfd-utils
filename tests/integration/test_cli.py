
import os
import subprocess
import pytest
from tests.helpers.generate_test_data import create_test_tfrecords

@pytest.fixture(scope="module")
def test_data():
    test_data_dir = "/tmp/tfd_utils_cli_test_data"
    num_files = 1
    records_per_file = 5
    create_test_tfrecords(test_data_dir, num_files, records_per_file)
    yield test_data_dir
    # Cleanup
    for file_idx in range(num_files):
        tfrecord_file = os.path.join(test_data_dir, f"test_data_{file_idx:03d}.tfrecord")
        index_file = os.path.splitext(tfrecord_file)[0] + ".index"
        if os.path.exists(tfrecord_file):
            os.remove(tfrecord_file)
        if os.path.exists(index_file):
            os.remove(index_file)
    if os.path.exists(test_data_dir):
        os.rmdir(test_data_dir)

def test_cli_list_features(test_data):
    tfrecord_file = os.path.join(test_data, "test_data_000.tfrecord")
    result = subprocess.run(["uv", "run", "tfd", "list", tfrecord_file], capture_output=True, text=True)
    
    assert result.returncode == 0
    output = result.stdout
    assert "Feature Name" in output
    assert "Type" in output
    assert "height" in output
    assert "key" in output
    assert "image" in output
    assert "metadata" in output
    assert "format" in output
    assert "width" in output

def test_cli_extract_record(test_data):
    tfrecord_file = os.path.join(test_data, "test_data_000.tfrecord")
    key = "test_000_0001"
    result = subprocess.run(["uv", "run", "tfd", "extract", tfrecord_file, key], capture_output=True, text=True)

    assert result.returncode == 0
    assert f"Saved image content from feature 'image'" in result.stdout
    assert f"Feature 'key[0]' (text): {key}" in result.stdout

    # Check if the image was created
    image_file = f"{key}_image_0.jpeg"
    assert os.path.exists(image_file)
    os.remove(image_file)


