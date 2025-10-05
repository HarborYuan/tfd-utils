from tfd_utils.random_access import TFRecordRandomAccess
from tfd_utils.pb2 import Example

# Initialize with a single file
reader = TFRecordRandomAccess("data.tfrecord")

# Get the total number of records
print(f"Total records: {len(reader)}")

# Access a record by key
key = "record_3"
if key in reader:
    example = reader[key]
    image_bytes = example.features.feature['image'].bytes_list.value[0]
    print(f"Successfully read image for key '{key}': {image_bytes.hex()}")

# You can also get a specific feature directly
image_bytes_direct = reader.get_feature("record_5", "image")
print(f"Successfully read image directly for key 'record_5': {image_bytes_direct.hex()}")

# Example with multiple files
# First, create another file
from tfd_utils.writer import TFRecordWriter
from tfd_utils.pb2 import Features, Feature, BytesList

with TFRecordWriter("data_2.tfrecord") as writer:
    key = b'record_10'
    example = Example(features=Features(feature={
        'key': Feature(bytes_list=BytesList(value=[key])),
        'image': Feature(bytes_list=BytesList(value=[b'new_image']))
    }))
    writer.write(example.SerializeToString())

# Now, read from multiple files
multi_reader = TFRecordRandomAccess(["data.tfrecord", "data_2.tfrecord"])
print(f"Total records in multiple files: {len(multi_reader)}")
record_10_bytes = multi_reader["record_10"]
example_10 = Example()
example_10.ParseFromString(record_10_bytes)
print(f"Image from second file: {example_10.features.feature['image'].bytes_list.value[0]}")
