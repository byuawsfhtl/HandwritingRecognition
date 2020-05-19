import tensorflow as tf
import os
from PIL import Image
from util.resize import resize_img


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
        img = Image.open(path)
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
