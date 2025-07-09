"""
Generate test TFRecord data for testing the random access functionality.
"""

import os
import requests
from PIL import Image
import io
import json
from typing import List, Dict, Any


from tfd_utils.pb2 import Example, Features, Feature, BytesList, Int64List, FloatList
from tfd_utils.writer import TFRecordWriter

def download_test_image(url: str = "https://yuanhaobo.me/assets/img/yuanhaobo.jpg?18dff0f3b5aca4712c52789805459350") -> bytes:
    """Download the test image and return as bytes."""
    print(f"Downloading test image from: {url}")
    response = requests.get(url)
    response.raise_for_status()
    return response.content


def create_test_example(key: str, image_bytes: bytes, metadata: Dict[str, Any]) -> Example:
    """Create a Example with the given data."""
    # Create features
    features = {
        'key': Feature(bytes_list=BytesList(value=[key.encode('utf-8')])),
        'image': Feature(bytes_list=BytesList(value=[image_bytes])),
        'metadata': Feature(bytes_list=BytesList(value=[json.dumps(metadata).encode('utf-8')])),
        'width': Feature(int64_list=Int64List(value=[metadata.get('width', 0)])),
        'height': Feature(int64_list=Int64List(value=[metadata.get('height', 0)])),
        'format': Feature(bytes_list=BytesList(value=[metadata.get('format', 'JPEG').encode('utf-8')])),
    }
    
    # Create example
    example = Example(features=Features(feature=features))
    return example


def create_test_tfrecords(output_dir: str, num_files: int = 3, records_per_file: int = 10) -> List[str]:
    """Create test TFRecord files with fake data."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Download the test image
    image_bytes = download_test_image()
    
    # Get image dimensions
    image = Image.open(io.BytesIO(image_bytes))
    width, height = image.size
    format_name = image.format
    
    tfrecord_files = []
    
    for file_idx in range(num_files):
        output_file = os.path.join(output_dir, f"test_data_{file_idx:03d}.tfrecord")
        tfrecord_files.append(output_file)
        
        print(f"Creating {output_file} with {records_per_file} records...")
        
        with TFRecordWriter(output_file) as writer:
            for record_idx in range(records_per_file):
                # Create unique key
                key = f"test_{file_idx:03d}_{record_idx:04d}"
                
                # Create metadata
                metadata = {
                    'width': width,
                    'height': height,
                    'format': format_name,
                    'file_index': file_idx,
                    'record_index': record_idx,
                    'description': f"Test image record {record_idx} from file {file_idx}",
                    'tags': ['test', 'sample', f'file_{file_idx}'],
                    'timestamp': f"2025-01-01T{file_idx:02d}:{record_idx:02d}:00Z"
                }
                
                # Create example
                example = create_test_example(key, image_bytes, metadata)
                
                # Write to file
                writer.write(example.SerializeToString())
        
        print(f"  Created {output_file}")
    
    print(f"\nTest data generation complete!")
    print(f"Created {len(tfrecord_files)} files with {num_files * records_per_file} total records")
    return tfrecord_files


def create_test_data_with_different_key_types(output_dir: str) -> str:
    """Create a TFRecord file with different key types for testing."""
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "test_different_keys.tfrecord")
    image_bytes = download_test_image()
    
    print(f"Creating {output_file} with different key types...")
    
    with TFRecordWriter(output_file) as writer:
        # String key
        example1 = Example(features=Features(feature={
            'key': Feature(bytes_list=BytesList(value=[b'string_key_001'])),
            'id': Feature(int64_list=Int64List(value=[1])),
            'score': Feature(float_list=FloatList(value=[1.5])),
            'image': Feature(bytes_list=BytesList(value=[image_bytes])),
            'type': Feature(bytes_list=BytesList(value=[b'string_key'])),
        }))
        writer.write(example1.SerializeToString())
        
        # Integer key
        example2 = Example(features=Features(feature={
            'key': Feature(bytes_list=BytesList(value=[b'int_key_002'])),
            'id': Feature(int64_list=Int64List(value=[2])),
            'score': Feature(float_list=FloatList(value=[42.0])),
            'image': Feature(bytes_list=BytesList(value=[image_bytes])),
            'type': Feature(bytes_list=BytesList(value=[b'int_key'])),
        }))
        writer.write(example2.SerializeToString())
        
        # Float key
        example3 = Example(features=Features(feature={
            'key': Feature(bytes_list=BytesList(value=[b'float_key_003'])),
            'id': Feature(int64_list=Int64List(value=[3])),
            'score': Feature(float_list=FloatList(value=[99.5])),
            'image': Feature(bytes_list=BytesList(value=[image_bytes])),
            'type': Feature(bytes_list=BytesList(value=[b'float_key'])),
        }))
        writer.write(example3.SerializeToString())
    
    print(f"  Created {output_file}")
    return output_file


if __name__ == "__main__":
    # Create test data directory
    test_data_dir = "/tmp/tfd_utils_test_data"
    
    # Create main test data
    tfrecord_files = create_test_tfrecords(test_data_dir, num_files=3, records_per_file=10)
    
    # Create test data with different key types
    different_keys_file = create_test_data_with_different_key_types(test_data_dir)
    
    print(f"\nTest data created in: {test_data_dir}")
    print("Files created:")
    for file in tfrecord_files + [different_keys_file]:
        print(f"  {file}")
