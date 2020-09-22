import os

import tensorflow as tf
import pandas as pd

# Default character set
DEFAULT_INDICES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                   28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
                   53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
                   78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101,
                   102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121,
                   122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141,
                   142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161,
                   162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181,
                   182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196]
DEFAULT_CHARS = [' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3',
                 '4', '5', '6', '7', '8', '9', ':', ';', '=', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', ']', '_', '`',
                 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                 'u', 'v', 'w', 'x', 'y', 'z', '|', '~', '£', '§', '¨', '«', '¬', '\xad', '°', '²', '´', '·', 'º', '»',
                 '¼', '½', '¾', 'À', 'Â', 'Ä', 'Ç', 'È', 'É', 'Ê', 'Ô', 'Ö', 'Ü', 'ß', 'à', 'á', 'â', 'ä', 'æ', 'ç',
                 'è', 'é', 'ê', 'ë', 'ì', 'î', 'ï', 'ñ', 'ò', 'ó', 'ô', 'ö', 'ø', 'ù', 'ú', 'û', 'ü', 'ÿ', 'ł', 'Œ',
                 'œ', 'Γ', 'Ζ', 'Τ', 'ά', 'ή', 'α', 'δ', 'ε', 'η', 'ι', 'κ', 'λ', 'μ', 'ν', 'ξ', 'ο', 'π', 'ρ', 'τ',
                 'υ', 'χ', 'ψ', 'ω', 'ό', 'ώ', 'І', '‒', '–', '—', '†', '‡', '‰', '‹', '›', '₂', '₤', '℔', '⅓', '⅔',
                 '⅕', '⅖', '⅗', '⅘', '⅙', '⅚', '⅛', '∆', '∇', '∫', '≠', '□', '♀', '♂', '✓', 'ｆ']


def read_charset(filename):
    """
    read_charset

    Read the indices and chars from a file. Not yet implemented...

    :param filename:
    :return: list of indices and chars
    """
    return [], []


def get_char2idx(filename=None):
    """
    get_char2idx

    A tensorflow lookup table is created and returned which allows us to encode word transcriptions on the fly
    in the tf.data api. A standard python dictionary won't work when tensorflow is running in graph mode. This
    function will return a lookup table to convert between chars and indices.

    :param filename: (Optional) filename that includes the desired character set
    :return: A tensorflow lookup table to convert characters to integers
    """
    if filename is None:
        indices = DEFAULT_INDICES
        chars = DEFAULT_CHARS
    else:
        indices, chars = read_charset(filename)

    char2idx = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(chars, dtype=tf.string),
            values=tf.constant(indices, dtype=tf.int64)
        ),
        default_value=0,
        name='char2idx_lookup'
    )

    return char2idx


def get_idx2char(filename=None):
    """
    get_idx2char

    A tensorflow lookup table is created and returned which allows us to encode word transcriptions on the fly
    in the tf.data api. A standard python dictionary won't work when tensorflow is running in graph mode. This
    function will return a lookup table to convert between indices and chars.

    :param filename:
    :return: A tensorflow lookup table to convert integers to characters
    """
    if filename is None:
        indices = DEFAULT_INDICES
        chars = DEFAULT_CHARS
    else:
        indices, chars = read_charset(filename)

    idx2char = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(indices, dtype=tf.int64),
            values=tf.constant(chars, dtype=tf.string)
        ),
        default_value='',
        name='idx2char_lookup'
    )

    return idx2char


def pad_or_truncate(t, sequence_size=128):
    """
    pad_or_trunc

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
    merge_repeating_values

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
    str_to_idxs

    Perform the actual lookup to convert a string to its integer representation. This function also performs
    padding according to the given sequence size. Works for use in the tf.data api in graph mode.

    :param string: The string to be converted
    :param char2idx: The tf lookup table
    :param sequence_size: The final sequence length
    :return: The converted string now in its integer representation
    """
    idxs = tf.map_fn(lambda char: char2idx.lookup(char), tf.strings.bytes_split(string), fn_output_signature=tf.int64)
    return pad_or_truncate(idxs, sequence_size=sequence_size)


def idxs_to_str(idxs, idx2char, merge_repeated=True):
    """
    idxs_to_str

    Perform the actual lookup to convert an integer to its string representation.
    Works for use in the tf.data api in graph mode.

    :param idxs: The idxs to be converted
    :param idx2char: The tf lookup table
    :param merge_repeated: Bool indicating whether or not to merge repeating values in the idx tensor
    :return: The converted idxs now in its string representation
    """
    if merge_repeated:
        idxs = merge_repeating_values(idxs)

    string = tf.map_fn(lambda idx: idx2char.lookup(idx), idxs, fn_output_signature=tf.string)
    string = tf.strings.reduce_join(string)
    return tf.strings.strip(string)


def str_to_idxs_batch(batch, char2idx, sequence_size=128):
    """
    str_to_idxs_batch

    Perform the same function as str_to_idxs, except a batch of strings are given as input

    :param batch: A batch of strings as tensor, list, or numpy array
    :param char2idx: The tf lookup table
    :param sequence_size: The final sequence length of each string
    :return: The converted strings now in its integer representation
    """
    return tf.map_fn(lambda string: str_to_idxs(string, char2idx, sequence_size=sequence_size), batch,
                     fn_output_signature=tf.int64)


def idxs_to_str_batch(batch, idx2char, merge_repeated=True):
    """
    idxs_to_str_batch

    Perform the same function as idxs_to_str, except a batch of idxs are given as input

    :param batch: A batch of idxs as tensor, list, or numpy array
    :param idx2char: The tf lookup table
    :param merge_repeated: Bool indicating whether or not to merge repeating values in the idx tensor
    :return: The converted idxs now in its string representation
    """
    return tf.map_fn(lambda idxs: idxs_to_str(idxs, idx2char, merge_repeated=merge_repeated), batch,
                     fn_output_signature=tf.string)


def read_and_encode_image(img_path, img_size=(64, 1024)):
    """
    read_and_encode_image

    Used by both encode_img_and_transcription (training) and encode_img_with_name (inference). This method
    simply loads the image given a file path and performs the necessary encoding/resizing/transposing that
    is necessary for use on the recognition model.

    :param img_path: The path to the desired image
    :param img_size: The size of the image after resizing/padding
    :return: The encoded image in its tensor/integer representation
    """
    img_bytes = tf.io.read_file(img_path)
    img = tf.image.decode_image(img_bytes, dtype=tf.float32)
    img = tf.image.per_image_standardization(img)
    img = tf.image.resize_with_pad(img, img_size[0], img_size[1])
    img = tf.expand_dims(tf.transpose(tf.squeeze(img)), 2)  # Transpose with channels last

    return img


def encode_img_and_transcription(img_path, transcription, char2idx, sequence_size=128, img_size: tuple = (64, 1024)):
    """
    encode_img_and_transcription

    The actual function to map image paths and string transcriptions to its tensor/integer representation.

    :param img_path: The path to the desired image
    :param transcription: The transcription of the image in integer form
    :param char2idx: The tf lookup table
    :param sequence_size: The final sequence length for transcriptions
    :param img_size: The size of the image after resizing/padding
    :return: The image and transcription in their tensor/integer representations.
    """
    img = read_and_encode_image(img_path, img_size)
    line = str_to_idxs(transcription, char2idx, sequence_size)
    return img, line


def encode_img_with_name(img_path, file_separator, img_size=(64, 1024)):
    """
    encode img_with_name

    Used to map img_paths to encoded images for inference. Returned is the encoded image and image name.

    :param img_path: The file path to the image
    :param file_separator: The os specific file separator - can be obtained with os.path.sep
    :param img_size: The size of the image after resizing/padding
    :return: The encoded image and image name
    """
    img = read_and_encode_image(img_path, img_size)
    img_name = tf.strings.split(tf.strings.split(img_path, sep=file_separator)[-1], sep='.')[0]

    return img, img_name


def get_dataset_size(csv_path):
    """
    get_dataset_size

    The tf.data api has a hard time producing the the dataset size. The cardinality() method often
    returns unknown even with the CsvDataset. This function uses pandas to get the length.

    :param csv_path: The path to csv containing information about the dataset
    :return: The size of the dataset
    """
    return len(pd.read_csv(csv_path, sep='\t', header=None, names=['img_path', 'transcription']))


def get_encoded_dataset_from_csv(csv_path, char2idx, max_seq_size, img_size):
    """
    get_encoded_dataset_from_csv

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
    return tf.data.experimental.CsvDataset(csv_path, ['img', 'trans'], field_delim='\t', use_quote_delim=False).map(
        lambda img_path, transcription: encode_img_and_transcription(
            tf.strings.join([path_prefix, tf.strings.reduce_join(tf.strings.split(img_path, '/'), separator=path_sep)],
                            separator=path_sep),
            transcription, char2idx, max_seq_size, img_size))


def get_encoded_inference_dataset_from_img_path(img_path, img_size):
    """
    get_encoded_inference_dataset_from_img_path

    Using the tf.data api, load all images from the desired path and return a dataset containing encoded images
    and the image name (without path or extension information).

    :param img_path: The path to the directory containing images
    :param img_size: The size of the image after resizing/padding (height, width)
    :return: The tf dataset containing encoded images and their respective string names
    """
    return tf.data.Dataset.list_files(img_path + '/*', shuffle=False).map(
        lambda path: encode_img_with_name(path, os.path.sep, img_size))
