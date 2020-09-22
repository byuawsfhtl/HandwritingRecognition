import tensorflow as tf


@tf.function
def model_inference(model, imgs):
    """
    model_inference

    Runs images through the model wrapped in a @tf.function annotation. This forces TensorFlow to run in Graph mode
    and speed up computation. This function is handy when performing inference.

    :param model: The handwriting recognition model
    :param imgs: A mini-batch of images
    :return: The output of the recognition model
    """
    return model(imgs, training=False)
