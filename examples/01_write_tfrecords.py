from tfd_utils.writer import TFRecordWriter
from tfd_utils.pb2 import Example, Features, Feature, BytesList

# Create a dummy image and label
image_bytes = b'\xDE\xAD\xBE\xEF'
label = b'cat'

with TFRecordWriter("data.tfrecord") as writer:
    for i in range(10):
        key = f'record_{i}'.encode('utf-8')
        example = Example(features=Features(feature={
            'key': Feature(bytes_list=BytesList(value=[key])),
            'image': Feature(bytes_list=BytesList(value=[image_bytes])),
            'label': Feature(bytes_list=BytesList(value=[label]))
        }))
        writer.write(example.SerializeToString())

print("Successfully wrote 10 records to data.tfrecord")
