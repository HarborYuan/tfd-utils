import tensorflow as tf

# Ensure you have run 01_write_tfrecords.py first to create this file
file_path = "data.tfrecord"

def parse_tfrecord_fn(record):
    feature_description = {
        'key': tf.io.FixedLenFeature([], tf.string),
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(record, feature_description)
    return example

dataset = tf.data.TFRecordDataset(file_path)
parsed_dataset = dataset.map(parse_tfrecord_fn)

print("Reading records with TensorFlow:")
for parsed_record in parsed_dataset.take(5):
    key = parsed_record['key'].numpy().decode('utf-8')
    image_bytes = parsed_record['image'].numpy()
    label = parsed_record['label'].numpy().decode('utf-8')
    print(f"  Key: {key}, Image: {image_bytes.hex()}, Label: {label}")
