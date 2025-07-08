#!/usr/bin/env python3
"""
Integration test for TFRecord Random Access functionality.
This script can be run independently to test the basic functionality.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tfd_utils.random_access import TFRecordRandomAccess
from tests.generate_test_data import create_test_tfrecords


def test_basic_functionality():
    """Test basic TFRecord random access functionality."""
    print("=== TFRecord Random Access Integration Test ===\n")
    
    # Create temporary directory for test data
    test_dir = tempfile.mkdtemp(prefix="tfd_utils_integration_test_")
    print(f"Test directory: {test_dir}")
    
    try:
        # Step 1: Generate test data
        print("\n1. Generating test data...")
        tfrecord_files = create_test_tfrecords(test_dir, num_files=2, records_per_file=5)
        print(f"   Created {len(tfrecord_files)} TFRecord files")
        
        # Step 2: Initialize random access reader
        print("\n2. Initializing TFRecord random access reader...")
        reader = TFRecordRandomAccess(tfrecord_files)
        print(f"   Reader initialized with {len(reader.tfrecord_files)} files")
        
        # Step 3: Test index building
        print("\n3. Building index...")
        index = reader.index
        print(f"   Index built with {len(index)} records")
        
        # Step 4: Test basic operations
        print("\n4. Testing basic operations...")
        
        # Get all keys
        keys = reader.get_keys()
        print(f"   Total keys: {len(keys)}")
        print(f"   Sample  (10): {keys[:10]}")  # Show first 10 keys
        
        # Test key existence
        test_key = keys[0]
        print(f"   Key '{test_key}' exists: {test_key in reader}")
        
        # Get a record
        record = reader.get_record(test_key)
        print(f"   Retrieved record for key '{test_key}': {record is not None}")
        
        if record:
            # Extract features
            key_feature = record.features.feature['key'].bytes_list.value[0].decode('utf-8')
            print(f"   Record key feature: {key_feature}")
            
            # Get image feature
            image_bytes = reader.get_feature(test_key, 'image')
            print(f"   Image data size: {len(image_bytes)} bytes")
            
            # Get metadata
            metadata_bytes = reader.get_feature(test_key, 'metadata')
            import json
            metadata = json.loads(metadata_bytes.decode('utf-8'))
            print(f"   Metadata: {metadata}")
        
        # Step 5: Test statistics
        print("\n5. Getting statistics...")
        stats = reader.get_stats()
        print(f"   Total records: {stats['total_records']}")
        print(f"   Total files: {stats['total_files']}")
        print(f"   Records per file: {stats['records_per_file']}")
        print(f"   Index file: {stats['index_file']}")
        
        # Step 6: Test different initialization methods
        print("\n6. Testing different initialization methods...")
        
        # Test with single file
        reader_single = TFRecordRandomAccess(tfrecord_files[0])
        print(f"   Single file reader: {len(reader_single)} records")
        
        # Test with glob pattern
        pattern = os.path.join(test_dir, "test_data_*.tfrecord")
        reader_glob = TFRecordRandomAccess(pattern)
        print(f"   Glob pattern reader: {len(reader_glob)} records")
        
        # Step 7: Test rebuild index
        print("\n7. Testing index rebuild...")
        original_index_file = reader.index_file
        reader.rebuild_index()
        print(f"   Index rebuilt successfully")
        print(f"   Index file still exists: {os.path.exists(original_index_file)}")
        
        print("\n=== All tests passed! ===")
        
    finally:
        # Cleanup
        print(f"\nCleaning up test directory: {test_dir}")
        shutil.rmtree(test_dir)


if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)
