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
def bp_decode(output):
    """
    Best-path decoding from raw model output.

    :param output: The model output, shape: (batch x sequence x classes)
    :return: The output of the best-path decoding
    """
    return tf.argmax(output, axis=2, output_type=tf.int32)


@tf.function
def wbs_decode(wbs, output):
    """
    Word-beam-search decoding from raw model output.

    :param wbs: The word beam search decoder object
    :param output: The model output, shape: (batch x sequence x classes)
    :return: The output of word beam search decoding
    """
    return wbs(output)


def prediction_confidence_by_word(output):
    """
    Given the output of the model, perform a best path decoding and provide a confidence score per word.

    :param output: The raw model output
    :return: A list of pairs of best-path decoded sequences along with its confidence score
             Example: [[[0, 5, 5, 2], .3739], [[3, 3, 1, 1, 1, 0, 7], .6891]]
    """
    label = tf.argmax(output, axis=2, output_type=tf.int32)
    space_indices = tf.where(label == 1)[:, 1]
    space_indices = tf.concat((space_indices, [tf.shape(output)[1]]), axis=0)  # Add index at the end of the sequence

    prev_index = 0

    prediction_confidence_list = list()

    for index in space_indices:
        word_slice = label[0][prev_index:index]
        raw_word_slice = output[:, prev_index:index, :]
        confidence = prediction_confidence(raw_word_slice)

        prediction_confidence_list.append([word_slice, confidence])

        prev_index = index + 1

    return prediction_confidence_list


def prediction_confidence(output):
    """
    Given the model output, give a confidence score for best path prediction

    :param output: The model's output, shape: (batch x sequence x num_classes)
    :return: The confidence score
    """
    batch_size = output.shape[0]
    seq_size = output.shape[1]

    values = merge_repeating_values(tf.squeeze(tf.argmax(output, 2, output_type=tf.int32), 0))
    mask = tf.not_equal(values, tf.constant(0, dtype=tf.int32))
    unpadded_label = tf.boolean_mask(values, mask)
    label = tf.expand_dims(pad_or_truncate(unpadded_label, sequence_size=seq_size), 0)

    input_lengths = tf.constant(np.full(batch_size, seq_size))
    label_lengths = tf.math.count_nonzero(label, axis=1)

    loss = tf.nn.ctc_loss(label, output, label_lengths, input_lengths, logits_time_major=False)

    probability = tf.exp(-loss)[0].numpy()

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
