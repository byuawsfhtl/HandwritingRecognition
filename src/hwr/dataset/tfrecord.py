import tensorflow as tf


def create_tfrecord_from_sequence(sequence, tfrecord_path):
    """
    Create a TfRecord dataset from a sequence

    :param sequence: The Keras sequence to load data of arbitrary format
    :param tfrecord_path: Filepath and name for location of TfRecord dataset
    """
    tf.print('Started creating TFRecord Dataset...')

    writer = tf.io.TFRecordWriter(tfrecord_path)

    for index, (img, label) in enumerate(sequence):
        feature = {'label': _bytes_feature(tf.io.serialize_tensor(label)),
                   'image': _bytes_feature(tf.io.serialize_tensor(img))}

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
        if index % 1000 == 0:
            tf.print(str(index) + '/' + str(len(sequence)))

    tf.print(str(len(sequence)) + '/' + str(len(sequence)))
    tf.print('Finished: TFRecord created at', tfrecord_path)


def read_tfrecord(single_record):
    """
    Function to decode a TfRecord. Usually this function will be called within
    a TfDataset map function. Note that out_types for image and label must be
    tf.float32 and tf.int64 respectively.

    :param single_record: A single TfRecord
    :return: A decoded image and label as tensors
    """
    feature_description = {
        'label': tf.io.FixedLenFeature((), tf.string),
        'image': tf.io.FixedLenFeature((), tf.string)
    }

    single_record = tf.io.parse_single_example(single_record, feature_description)

    image = tf.io.parse_tensor(single_record['image'], out_type=tf.float32)
    label = tf.io.parse_tensor(single_record['label'], out_type=tf.int64)

    return image, label


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
