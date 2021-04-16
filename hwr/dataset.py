import os

import csv
import tensorflow as tf
import pandas as pd

# The default list of characters used in the recognition model
DEFAULT_CHARS = ' !"#$%&\'()*+,-./0123456789:;=?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_`abcdefghijklmnopqrstuvwxyz|~£§¨«¬\xad' \
                '°²´·º»¼½¾ÀÂÄÇÈÉÊÔÖÜßàáâäæçèéêëìîïñòóôöøùúûüÿłŒœΓΖΤάήαδεηικλμνξοπρτυχψωόώІ‒–—†‡‰‹›₂₤℔⅓⅔⅕⅖⅗⅘⅙⅚⅛∆∇∫≠□♀♂✓ｆ'
# The default list of non-punctuation characters needed for the word beam search decoding algorithm
DEFAULT_NON_PUNCTUATION = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÀÂÄÇÈÉÊÔÖÜßàáâäæçèéêëìîïñòóôöøùúûüÿ' \
                          'łŒœΓΖΤάήαδεηικλμνξοπρτυχψωόώІ'
# The default list of punctuation characters needed for hte word beam search decoding algorithm
DEFAULT_PUNCTUATION = ' !"#$%&\'()*+,-./0123456789:;=?[]_`|~£§¨«¬°²´·º»¼½¾‒–—†‡‰‹›₂₤℔⅓⅔⅕⅖⅗⅘⅙⅚⅛∆∇∫≠□♀♂✓'


def str_charset_to_lists(charset):
    """
    Turns string containing all desired characters into list of chars and indices. This is required for mapping
    between integer and char representations for use in the recognition model.

    :param charset: charset as string of chars to be represented in model.
    """
    chars = list(charset)
    indices = list(range(1, len(chars) + 1))
    return chars, indices


def get_char2idx(charset):
    """
    A tensorflow lookup table is created and returned which allows us to encode word transcriptions on the fly
    in the tf.data api. A standard python dictionary won't work when tensorflow is running in graph mode. This
    function will return a lookup table to convert between chars and indices.

    :param charset: string containing all desired characters to be represented
    :return: A tensorflow lookup table to convert characters to integers
    """
    chars, indices = str_charset_to_lists(charset)

    char2idx = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(chars, dtype=tf.string),
            values=tf.constant(indices, dtype=tf.int64),
            key_dtype=tf.string,
            value_dtype=tf.int64
        ),
        default_value=0,
        name='char2idx_lookup'
    )

    return char2idx


def get_idx2char(charset):
    """
    A tensorflow lookup table is created and returned which allows us to encode word transcriptions on the fly
    in the tf.data api. A standard python dictionary won't work when tensorflow is running in graph mode. This
    function will return a lookup table to convert between indices and chars.

    :param charset: string containing all desired characters to be represented.
    :return: A tensorflow lookup table to convert integers to characters
    """
    chars, indices = str_charset_to_lists(charset)

    idx2char = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(indices, dtype=tf.int64),
            values=tf.constant(chars, dtype=tf.string),
            key_dtype=tf.int64,
            value_dtype=tf.string
        ),
        default_value='',
        name='idx2char_lookup'
    )

    return idx2char


def pad_or_truncate(t, sequence_size=128):
    """
    Pad or truncate a tensor to a fixed sequence length. Works for use in the tf.data api in graph mode.

    :param t: The tensor to pad or truncate
    :param sequence_size: The final sequence length of the tensor
    :return:
    """
    dim = tf.size(t)
    return tf.cond(tf.equal(dim, sequence_size), lambda: t,
                   lambda: tf.cond(tf.greater(dim, sequence_size), lambda: tf.slice(t, [0], [sequence_size]),
                                   lambda: tf.concat([t, tf.zeros(sequence_size - dim, dtype=tf.int64)], 0)))


def merge_repeating_values(t):
    """
    Merge repeating indices/characters in a tensor. Utilizes only tf.* functions which makes it
    usable in graph mode.

    :param t: The tensor to have repeated indices/characters merged
    :return: A new tensor with repeating values merged
    """
    t2 = tf.roll(tf.pad(t, [[0, 1]], constant_values=-1), -1, 0)[:tf.size(t)]
    not_equal = tf.math.not_equal(t, t2)
    indices = tf.where(not_equal)
    return tf.reshape(tf.gather(t, indices), [-1])


def str_to_idxs(string, char2idx, sequence_size):
    """
    Perform the actual lookup to convert a string to its integer representation. This function also performs
    padding according to the given sequence size. Works for use in the tf.data api in graph mode.

    :param string: The string to be converted
    :param char2idx: The tf lookup table
    :param sequence_size: The final sequence length
    :return: The converted string now in its integer representation
    """
    idxs = tf.map_fn(lambda char: char2idx.lookup(char), tf.strings.unicode_split(string, 'UTF-8'), dtype=tf.int64)
    return pad_or_truncate(idxs, sequence_size=sequence_size)


def idxs_to_str(idxs, idx2char, merge_repeated=True):
    """
    Perform the actual lookup to convert an integer to its string representation.
    Works for use in the tf.data api in graph mode.

    :param idxs: The idxs to be converted
    :param idx2char: The tf lookup table
    :param merge_repeated: Bool indicating whether or not to merge repeating values in the idx tensor
    :return: The converted idxs now in its string representation
    """
    if merge_repeated:
        idxs = merge_repeating_values(idxs)

    string = tf.map_fn(lambda idx: idx2char.lookup(idx), idxs, dtype=tf.string)
    string = tf.strings.reduce_join(string)
    return tf.strings.strip(string)


def str_to_idxs_batch(batch, char2idx, sequence_size=128):
    """
    Perform the same function as str_to_idxs, except a batch of strings are given as input

    :param batch: A batch of strings as tensor, list, or numpy array
    :param char2idx: The tf lookup table
    :param sequence_size: The final sequence length of each string
    :return: The converted strings now in its integer representation
    """
    return tf.map_fn(lambda string: str_to_idxs(string, char2idx, sequence_size=sequence_size), batch,
                     dtype=tf.int64)


def idxs_to_str_batch(batch, idx2char, merge_repeated=True):
    """
    Perform the same function as idxs_to_str, except a batch of idxs are given as input

    :param batch: A batch of idxs as tensor, list, or numpy array
    :param idx2char: The tf lookup table
    :param merge_repeated: Bool indicating whether or not to merge repeating values in the idx tensor
    :return: The converted idxs now in its string representation
    """
    return tf.map_fn(lambda idxs: idxs_to_str(idxs, idx2char, merge_repeated=merge_repeated), batch,
                     dtype=tf.string)


def img_resize_with_pad(img_tensor, desired_size, pad_value=255):
    """
    The standard tf.image.resize_with_pad function does not allow for specifying the pad value,
    so we create a function with that capability here. Aspect ratio will be preserved.

    :param img_tensor: The image tensor to be resized and padded
    :param desired_size: The desired size (height, width)
    :param pad_value: The value to pad the tensor with
    """
    img_size = tf.shape(img_tensor)

    img_ratio = img_size[0] / img_size[1]
    desired_ratio = desired_size[0] / desired_size[1]

    if img_ratio >= desired_ratio:
        # Solve by height
        new_height = desired_size[0]
        new_width = int(desired_size[0] // img_ratio)
    else:
        new_height = int(desired_size[1] * img_ratio)
        new_width = desired_size[1]
        # Solve by width

    resized_img = tf.image.resize(img_tensor, (new_height, new_width), method=tf.image.ResizeMethod.BICUBIC)

    pad_height = desired_size[0] - new_height
    pad_width = desired_size[1] - new_width

    img_padded = tf.pad(resized_img, [[pad_height, 0], [0, pad_width], [0, 0]], constant_values=pad_value)

    return img_padded


def read_and_encode_image(img_path, img_size=(64, 1024)):
    """
    Used by both encode_img_and_transcription (training) and encode_img_with_name (inference). This method
    simply loads the image given a file path and performs the necessary encoding/resizing/transposing that
    is necessary for use on the recognition model.

    :param img_path: The path to the desired image
    :param img_size: The size of the image after resizing/padding
    :return: The encoded image in its tensor/integer representation
    """
    img_bytes = tf.io.read_file(img_path)
    img = tf.image.decode_image(img_bytes, channels=1, expand_animations=False)
    img = img_resize_with_pad(img, img_size)
    img = tf.image.per_image_standardization(img)

    return img


def encode_img_and_transcription(img_path, transcription, char2idx, sequence_size=128, img_size: tuple = (64, 1024)):
    """
    The actual function to map image paths and string transcriptions to its tensor/integer representation.

    :param img_path: The path to the desired image
    :param transcription: The transcription of the image in integer form
    :param char2idx: The tf lookup table
    :param sequence_size: The final sequence length for transcriptions
    :param img_size: The size of the image after resizing/padding
    :return: The image and transcription in their tensor/integer representations.
    """
    img = read_and_encode_image(img_path, img_size=img_size)
    line = str_to_idxs(transcription, char2idx, sequence_size)
    return img, line


def encode_img_with_name(img_path, img_size=(64, 1024)):
    """
    Used to map img_paths to encoded images for inference. Returned is the encoded image and image name.

    :param img_path: The file path to the image
    :param img_size: The size of the image after resizing/padding
    :return: The encoded image and image path
    """
    img = read_and_encode_image(img_path, img_size)
    return img, img_path


def get_dataset_size(csv_path):
    """
    The tf.data api has a hard time producing the the dataset size. The cardinality() method often
    returns unknown even with the CsvDataset. This function uses pandas to get the length.

    :param csv_path: The path to csv containing information about the dataset
    :return: The size of the dataset
    """
    return len(pd.read_csv(csv_path, sep='\t', header=None, names=['img_path', 'transcription'], quoting=csv.QUOTE_NONE))


def get_encoded_dataset_from_csv(csv_path, char2idx, max_seq_size, img_size):
    """
    Using the tf.data api, load the desired csv with img_path and transcription data, encode the images and
    transcriptions for use on the recognition model and return the desired tf dataset.

    :param csv_path: The path to the tab delimited csv file containing | Image Path | Transcription |
    :param char2idx: The tf lookup table to map characters to their respective integer representation
    :param max_seq_size: The final sequence length for transcriptions
    :param img_size: The size of the image after resizing/padding (height, width).
    :return: The tf dataset containing encoded images and their respective transcriptions
    """
    path_sep = os.path.sep
    path_prefix = tf.strings.join(csv_path.split('/')[:-1], path_sep)
    return tf.data.experimental.CsvDataset(csv_path, ['img', 'trans'], field_delim='\t', use_quote_delim=False, na_value='').map(
        lambda img_path, transcription: encode_img_and_transcription(
            tf.strings.join([path_prefix, tf.strings.reduce_join(tf.strings.split(img_path, '/'), separator=path_sep)],
                            separator=path_sep),
            transcription, char2idx, max_seq_size, img_size),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)


def get_encoded_inference_dataset_from_img_path(img_path, img_size, include_subdirs=False):
    """
    Using the tf.data api, load all images from the desired path and return a dataset containing encoded images
    and the image name (without path or extension information).

    :param img_path: The path to the directory containing images
    :param img_size: The size of the image after resizing/padding (height, width)
    :param include_subdirs: Whether to include subdirectories of img_path, default is False
    :return: The tf dataset containing encoded images and their respective string names
    """
    if include_subdirs:
        # The first entry in each tuple returned from os.walk is the directory
        dirs = [os.path.join(dir_tuple[0], '*.*') for dir_tuple in os.walk(img_path)]
    else:
        dirs = os.path.join(img_path, '*.*')

    return tf.data.Dataset.list_files(dirs, shuffle=False).map(lambda path: encode_img_with_name(path, img_size),
                                                               num_parallel_calls=tf.data.experimental.AUTOTUNE)