import tensorflow as tf
import numpy as np

from hwr.dataset import merge_repeating_values, pad_or_truncate


@tf.function
def model_inference(model, imgs):
    """
    Runs the images through the model wrapped in a @tf.function annotation. This forces TensorFlow to run in Graph mode
    and speed up computation. It does not perform any sort of decoding and returns the raw model output.

    :param model: The handwriting recognition model
    :param imgs: A mini-batch of images
    :return: The output of the recognition model
    """
    return model(imgs, training=False)


@tf.function
def bp_decode(output, axis=2):
    """
    Best-path decoding from raw model output.

    :param output: The model output, shape: (batch x sequence x classes)
    :param axis: The axis to perform on the model output to perform best_path decoding
    :return: The output of the best-path decoding
    """
    return tf.argmax(output, axis=axis, output_type=tf.int64)


def merge_repeating_and_pad(single, seq_size):
    single = merge_repeating_values(single)
    single = tf.gather_nd(single, tf.where(single))
    single = pad_or_truncate(single, sequence_size=seq_size)
    return single


def prediction_confidence(output, prediction):
    """
    Given the model output, give a confidence score for the given prediction

    :param output: The model's output, shape: (batch x sequence x num_classes)
    :param prediction: The model's prediction, shape: (batch x sequence)
    :return: The confidence scores as tensor, shape(batch)
    """
    batch_size = output.shape[0]
    seq_size = output.shape[1]

    label = tf.map_fn(lambda single: merge_repeating_and_pad(single, seq_size), prediction)

    input_lengths = tf.constant(np.full(batch_size, seq_size))
    label_lengths = tf.math.count_nonzero(label, axis=1)

    loss = tf.nn.ctc_loss(label, output, label_lengths, input_lengths, logits_time_major=False)

    probability = tf.exp(-loss).numpy()

    return probability


def word_segments_from_sequence(prediction, space_index=1, buffer=3):
    """
    :param prediction: The model's prediction for the sequence
    :param space_index: The character index used for spaces
    :param buffer: The additional number of timesteps to be added to the end of a segment after splitting on spaces
    :return: Word Segments as list
    """
    max_length = tf.size(prediction).numpy()
    space_indices = tf.squeeze(tf.where(prediction == space_index)).numpy()

    # If only a single space is found, make sure space indices is iterable
    if tf.rank(space_indices) == 0:
        space_indices = tf.expand_dims(space_indices, 0).numpy()

    prev_index = 0

    segments = []
    for index in space_indices:
        if index - prev_index > 1:
            word_segment = prediction[prev_index:index]
            start, end = find_start_and_end_from_segment(word_segment, prev_index, buffer)
            segments.append([start, end])

        prev_index = index

    # Get the last word
    if max_length != prev_index + 1:
        word_segment = prediction[prev_index:max_length]
        start, end = find_start_and_end_from_segment(word_segment, prev_index, buffer)
        segments.append([start, end])

    return segments


def find_start_and_end_from_segment(word_segment, prev_index, buffer):
    """
    Given a word_segment, the previous word's index and the desired buffer, return the
    starting and ending indices of the word for the entire prediction sequence.

    :param word_segment: The model's prediction for a single word
    :param prev_index: The previous word's index
    :param buffer: The additional number of timesteps to be added to the end of a segment after splitting on spaces
    :return: The starting and ending indices
    """
    non_blank_indices = tf.where(word_segment != 0)

    minimum = prev_index + tf.reduce_min(non_blank_indices).numpy()
    maximum = prev_index + tf.reduce_max(non_blank_indices).numpy()

    return minimum, maximum + buffer
