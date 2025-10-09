
import argparse
import os
from .random_access import TFRecordRandomAccess
from .pb2 import Example

def list_keys(args):
    """List all keys in a TFRecord file."""
    reader = TFRecordRandomAccess(args.path)
    keys = reader.get_keys()
    for key in keys:
        print(key)

def extract_record(args):
    """Extract a single record and display or save it."""
    reader = TFRecordRandomAccess(args.path)
    record = reader.get_record(args.key)
    if not record:
        print(f"Record with key '{args.key}' not found.")
        return

    for feature_name, feature in record.features.feature.items():
        if feature.bytes_list.value:
            for i, data in enumerate(feature.bytes_list.value):
                # Try to detect image format from magic numbers
                image_format = None
                if data.startswith(b'\xff\xd8'):
                    image_format = 'jpeg'
                elif data.startswith(b'\x89PNG\r\n\x1a\n'):
                    image_format = 'png'
                elif data.startswith(b'GIF87a') or data.startswith(b'GIF89a'):
                    image_format = 'gif'

                if image_format:
                    output_filename = f"{args.key}_{feature_name}_{i}.{image_format}"
                    with open(output_filename, 'wb') as f:
                        f.write(data)
                    print(f"Saved image content from feature '{feature_name}' to {output_filename}")
                else:
                    # Not an image, try to decode as text
                    try:
                        text = data.decode('utf-8')
                        print(f"Feature '{feature_name}[{i}]' (text): {text}")
                    except UnicodeDecodeError:
                        print(f"Feature '{feature_name}[{i}]' (binary): {len(data)} bytes")

        elif feature.int64_list.value:
            print(f"Feature '{feature_name}' (int64): {list(feature.int64_list.value)}")
        elif feature.float_list.value:
            print(f"Feature '{feature_name}' (float): {list(feature.float_list.value)}")

def main():
    """Main entry point for the tfd-utils CLI."""
    parser = argparse.ArgumentParser(description="TFRecord Utilities CLI")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # 'list' command
    parser_list = subparsers.add_parser('list', help="List keys in a TFRecord file")
    parser_list.add_argument('path', help="Path to the TFRecord file or directory of files")
    parser_list.set_defaults(func=list_keys)

    # 'extract' command
    parser_extract = subparsers.add_parser('extract', help="Extract a record by key")
    parser_extract.add_argument('path', help="Path to the TFRecord file or directory of files")
    parser_extract.add_argument('key', help="Key of the record to extract")
    parser_extract.set_defaults(func=extract_record)

    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
