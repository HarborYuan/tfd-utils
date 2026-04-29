
import argparse
import os
import shutil
from pathlib import Path
from .random_access import TFRecordRandomAccess
from .pb2 import Example
from .tar_converter import convert_tars, resolve_input_paths

def list_features(args):
    """List feature names and their types of the first record in a TFRecord file."""
    reader = TFRecordRandomAccess(args.path)
    keys = reader.get_keys()
    if not keys:
        print("No records found.")
        return

    # Get the first record to inspect its features
    first_key = keys[0]
    record = reader.get_record(first_key)
    if not record:
        print(f"Could not read the first record (key: {first_key}).")
        return

    print(f"Features in the first record (key: {first_key}):")
    print("-" * 40)
    print(f"{'Feature Name':<20} {'Type':<10}")
    print("-" * 40)

    for feature_name, feature in record.features.feature.items():
        feature_type = "Unknown"
        if feature.bytes_list.value:
            feature_type = "bytes"
        elif feature.int64_list.value:
            feature_type = "int64"
        elif feature.float_list.value:
            feature_type = "float"
        
        print(f"{feature_name:<20} {feature_type:<10}")
    print("-" * 40)

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

def get_feature(args):
    """Extract and display a single feature from a record."""
    try:
        path, key, feature_name = args.spec.rsplit(':', 2)
    except ValueError:
        print("Invalid format for 'get' command. Expected format: path:key:feature_name", file=sys.stderr)
        sys.exit(1)

    reader = TFRecordRandomAccess(path)
    feature_value = reader.get_feature(key, feature_name)

    if feature_value is None:
        print(f"Feature '{feature_name}' not found for key '{key}'.", file=sys.stderr)
        sys.exit(1)

    if isinstance(feature_value, bytes):
        # Try to detect image format from magic numbers
        image_format = None
        if feature_value.startswith(b'\xff\xd8'):
            image_format = 'jpeg'
        elif feature_value.startswith(b'\x89PNG\r\n\x1a\n'):
            image_format = 'png'
        elif feature_value.startswith(b'GIF87a') or feature_value.startswith(b'GIF89a'):
            image_format = 'gif'

        if image_format:
            output_filename = f"{key}_{feature_name}_0.{image_format}"
            with open(output_filename, 'wb') as f:
                f.write(feature_value)
            print(f"Saved image content from feature '{feature_name}' to {output_filename}")
        else:
            # Not an image, try to decode as text
            try:
                text = feature_value.decode('utf-8')
                print(f"Feature '{feature_name}[0]' (text): {text}")
            except UnicodeDecodeError:
                print(f"Feature '{feature_name}[0]' (binary): {len(feature_value)} bytes")

    elif isinstance(feature_value, int):
        print(f"Feature '{feature_name}' (int64): [{feature_value}]")
    elif isinstance(feature_value, float):
        print(f"Feature '{feature_name}' (float): [{feature_value}]")

def install_skill(args):
    """Install the tfd-utils Claude Code skill to ~/.claude/skills/tfd-utils/."""
    from . import __version__
    src = Path(__file__).parent / "skills" / "SKILL.md"
    dest_dir = Path.home() / ".claude" / "skills" / "tfd-utils"
    dest = dest_dir / "SKILL.md"
    dest_dir.mkdir(parents=True, exist_ok=True)
    content = src.read_text(encoding="utf-8").replace("{{VERSION}}", __version__)
    dest.write_text(content, encoding="utf-8")
    print(f"Skill installed to {dest} (tfd-utils v{__version__})")
    print("Claude will now assist with tfd-utils in any project.")


def run_prebuild(args):
    """Pre-build .index files for one or more TFRecord files (no training needed)."""
    import sys
    import time
    import multiprocessing
    from .random_access import TFRecordRandomAccess

    workers = args.workers if args.workers is not None else max(multiprocessing.cpu_count() * 2, 32)
    print(f"Pre-building TFRecord index for: {args.path}")
    print(f"Using {workers} worker processes (override with --workers / -w).")
    t0 = time.time()
    reader = TFRecordRandomAccess(args.path, max_workers=workers)
    elapsed = time.time() - t0
    stats = reader.get_stats()
    print(
        f"Done in {elapsed:.1f}s. "
        f"files={stats.get('total_files', '?')}  records={stats.get('total_records', '?')}"
    )
    print("Subsequent training runs will reuse the cached .index files (mtime-checked).")


def run_convert(args):
    """Convert tar archive(s) to TFRecord files."""
    import sys
    paths = resolve_input_paths(args.input)
    if not paths:
        print(f"No tar files found for input: {args.input}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(paths)} tar file(s) to convert.")
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    convert_tars(paths, args.output_dir, args.delete, args.workers)


def main():
    """Main entry point for the tfd-utils CLI."""
    parser = argparse.ArgumentParser(description="TFRecord Utilities CLI")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # 'list' command
    parser_list = subparsers.add_parser('list', help="List feature names of the first record in a TFRecord file")
    parser_list.add_argument('path', help="Path to the TFRecord file or directory of files")
    parser_list.set_defaults(func=list_features)

    # 'extract' command
    parser_extract = subparsers.add_parser('extract', help="Extract a record by key")
    parser_extract.add_argument('path', help="Path to the TFRecord file or directory of files")
    parser_extract.add_argument('key', help="Key of the record to extract")
    parser_extract.set_defaults(func=extract_record)

    # 'get' command
    parser_get = subparsers.add_parser('get', help="Extract a single feature from a record by key")
    parser_get.add_argument('spec', help="Specification in the format 'path:key:feature_name'")
    parser_get.set_defaults(func=get_feature)

    # 'convert' command
    parser_convert = subparsers.add_parser(
        'convert',
        help="Convert tar archive(s) to TFRecord file(s)",
    )
    parser_convert.add_argument(
        'input',
        help="Tar file, directory of tar files, or glob pattern (e.g. '/data/sa1b/*.tar')",
    )
    parser_convert.add_argument(
        '--output-dir', '-o',
        default=None,
        help="Directory to write TFRecord files (default: same directory as each source tar)",
    )
    parser_convert.add_argument(
        '--delete', '-d',
        action='store_true',
        default=False,
        help="Delete each source tar after successful conversion",
    )
    parser_convert.add_argument(
        '--workers', '-w',
        type=int,
        default=16,
        help="Number of parallel worker processes (default: 16)",
    )
    parser_convert.set_defaults(func=run_convert)

    # 'install-skill' command
    parser_skill = subparsers.add_parser(
        'install-skill',
        help="Install the tfd-utils Claude Code skill to ~/.claude/skills/tfd-utils/",
    )
    parser_skill.set_defaults(func=install_skill)

    # 'prebuild' command
    parser_prebuild = subparsers.add_parser(
        'prebuild',
        help="Pre-build .index for TFRecord file(s) before training (avoids long startup hang)",
    )
    parser_prebuild.add_argument(
        'path',
        help="TFRecord file, directory, or glob (e.g. '/path/to/shards/*.tfrecord')",
    )
    parser_prebuild.add_argument(
        '--workers', '-w',
        type=int,
        default=None,
        help="Parallel worker processes (default: 2x CPU count, min 32)",
    )
    parser_prebuild.set_defaults(func=run_prebuild)

    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
