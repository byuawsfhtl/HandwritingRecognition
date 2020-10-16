import tensorflow as tf


@tf.function
def model_inference(model, imgs):
    """
    Runs images through the model wrapped in a @tf.function annotation. This forces TensorFlow to run in Graph mode
    and speed up computation. This function is handy when performing inference.

    :param model: The handwriting recognition model
    :param imgs: A mini-batch of images
    :return: The output of the recognition model
    """
    return model(imgs, training=False)


@tf.function
def model_inference_bp_wbs(model, imgs, wbs):
    """
    Runs images through teh model wrapped in a @tf.function annotation. This forces Tensorflow to run in Graph mode
    and speed up computation. It also performs word beam search and best path decoding and returns the output of those
    decoding algorithms.

    :param model: The handwriting recognition model
    :param imgs: A mini-batch of images
    :param wbs: The word beam search decoder object
    :return: The output of best path decoding, The output of word beam search decoding
    """
    output = model(imgs, training=False)
    bp_output = tf.argmax(output, 2)
    wbs_output = wbs(output)

    return bp_output, wbs_output
