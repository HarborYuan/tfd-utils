from tfd_utils.writer import TFRecordWriter
from tfd_utils.random_access import TFRecordRandomAccess
from tfd_utils.pb2 import Example, Features, Feature, BytesList

# --- Custom Key Feature ---

# Create a file with a custom ID feature
file_with_custom_key = "custom_key.tfrecord"
with TFRecordWriter(file_with_custom_key) as writer:
    for i in range(5):
        example = Example(features=Features(feature={
            'user_id': Feature(bytes_list=BytesList(value=[f'user_{i}'.encode('utf-8')])),
            'data': Feature(bytes_list=BytesList(value=[b'some_data']))
        }))
        writer.write(example.SerializeToString())

# Read using the custom key
reader_custom_key = TFRecordRandomAccess(file_with_custom_key, key_feature_name="user_id")

key = "user_3"
if key in reader_custom_key:
    record_bytes = reader_custom_key[key]
    example = Example()
    example.ParseFromString(record_bytes)
    data = example.features.feature['data'].bytes_list.value[0]
    print(f"Successfully read data for custom key '{key}': {data}")


# --- Custom Index Caching ---

# Specify a custom location for the index file
reader_custom_index = TFRecordRandomAccess(
    file_with_custom_key,
    key_feature_name="user_id",
    index_file="my_custom_index.cache"
)

print(f"Custom index file will be created at: {reader_custom_index.index_file}")
# Accessing a record will trigger index creation
_ = reader_custom_index["user_1"]


# You can also force the index to be rebuilt
print("Rebuilding index...")
reader_custom_index.rebuild_index()
print("Index rebuilt.")
