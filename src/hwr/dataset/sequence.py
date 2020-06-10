import tensorflow as tf
import pandas as pd
from PIL import Image
import os
import csv
from src.hwr.util.resize import resize_img
from src.hwr.util.encoder import Encoder


class InferenceSequence(tf.keras.utils.Sequence):
    """
    InferenceSequence

    Keras Sequence class for loading image data at inference time. Note that data can be loaded
    much quicker if TfRecord format is used.
    """

    def __init__(self, img_path, desired_size=(64, 1024)):
        """
        Setup necessary file paths and load csv data using pandas.

        :param img_path: The filepath to the images
        :param desired_size: The shape all images will be resized and padded to
        """
        if not os.path.exists(img_path):
            raise Exception('Image path does not exist in', img_path)

        self.img_path = img_path
        self.desired_size = desired_size
        self.img_list = os.listdir(img_path)

    def tensor_image(self, path):
        """
        Open an image file using PIL, resize it, and convert to tensor.

        :param path: Filepath to the image to be opened
        :return: The image as a tensor
        """
        img = Image.open(path).convert('L')
        img = resize_img(img, self.desired_size)
        x = tf.constant(img, dtype=tf.float32)

        return x

    def __getitem__(self, index):
        """
        The method to index into the sequence to grab images and labels in tensor format.

        :param index: The index of the image/label to be retrieved
        :return: image as tensor and image_name as string
        """
        img = self.tensor_image(os.path.join(self.img_path, self.img_list[index]))
        img = tf.transpose(img)
        img = tf.constant(img, dtype=tf.float32)
        img = tf.expand_dims(img, 2)
        img = tf.expand_dims(img, 0)

        return img, self.img_list[index]

    def __len__(self):
        """
        The length of the sequence.

        :return: sequence length
        """
        return len(self.img_list)


class TrainSequence(tf.keras.utils.Sequence):
    """
    TrainSequence

    Keras Sequence class for loading image and label data during training time. Note that data can be loaded
    much quicker if TfRecord format is used.
    """

    def __init__(self, img_path, label_path, desired_size=(64, 1024), encode_labels=True,
                 encode_max_seq_size=128, encode_char_set_path=None):
        """
        Setup necessary file paths and load csv data using pandas.

        :param img_path: The filepath to the images
        :param label_path: The filepath to the labels CSV
        :param desired_size: The shape all images will be resized and padded to
        :param encode_labels: T/F - Whether or not to receive labels as strings or indices
        :param encode_max_seq_size: Max sequence size as given in Encoder
        :param encode_char_set_path: Charset path as given in Encoder
        """
        if not os.path.exists(label_path):
            raise Exception('Label CSV not contained in', label_path)
        elif not os.path.exists(img_path):
            raise Exception('Image path does not exist in', img_path)

        if encode_labels:
            self.encoder = Encoder(encode_char_set_path, max_sequence_size=encode_max_seq_size)

        self.img_path = img_path
        self.desired_size = desired_size
        self.encode_labels = encode_labels
        self.df = pd.read_csv(label_path, header=None, sep='\t', names=['word', 'transcription'],
                              quoting=csv.QUOTE_NONE)

    def tensor_image(self, path):
        """
        Open an image file using PIL, resize it, and convert to tensor.

        :param path: Filepath to the image to be opened
        :return: The image as a tensor
        """
        img = Image.open(path).convert('L')
        img = resize_img(img, self.desired_size)
        x = tf.constant(img, dtype=tf.float32)

        return x

    def __getitem__(self, index):
        """
        The method to index into the sequence to grab images and labels in tensor format.

        :param index: The index of the image/label to be retrieved
        :return: image and label as tensors
        """
        img_name = self.df['word'][index]
        if '.' not in img_name:
            img_name += '.png'

        img = self.tensor_image(os.path.join(self.img_path, img_name))
        img = tf.transpose(img)
        img = tf.constant(img, dtype=tf.float32)
        img = tf.expand_dims(img, 2)

        transcription = str(self.df['transcription'][index])

        if self.encode_labels:
            label = self.encoder.str_to_idxs(transcription)
            label = tf.constant(label, dtype=tf.int64)
        else:
            label = tf.constant(transcription, dtype=tf.string)

        return img, label

    def __len__(self):
        """
        The length of the sequence.

        :return: sequence length
        """
        return len(self.df)
