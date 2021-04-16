import tensorflow as tf


def augment_batched_dataset(ds):
    """
    Maps the batches within a dataset to augment them.

    :param ds: the dataset to augment
    :return the augmented dataset
    """
    return ds.map(lambda img, trans : (augment_batch(img), trans))


def augment_batch(batch):
    """
    Randomly applies different augmentations to the given batch.

    :param batch: the batch to augment
    :return: the augmented batch
    """
    if tf.random.uniform([]) < 0.5:
        batch = double_batch_bleed_through(batch)
    if tf.random.uniform([]) < 0.5:
        batch = add_noise(batch)
    return batch


def add_noise(img):
    """
    Adds normally distributed noise to each element in the tensor.
    **Do not call per_image_standardization before this, as the stddev is hardcoded to 20**

    :param img: the image/batch to add noise to
    :return: the input image with noise added.
    """
    rand = tf.random.normal(tf.shape(img), mean=0.0, stddev=20.0, dtype=tf.dtypes.float32)
    return tf.math.add(img, rand)


def reverse_and_lighten(img):
    """
    Reverses the input line on the second dimension (x) and shifts the pixels multiplicatively
    closer to white to lighten it.

    :param img: the image/batch tensor to reverse and lighten
    :return: a reversed, lightened version on the input tensor
    """
    img = tf.reverse(img, [2]) # reverse on x dimension
    img = img - 255 # shift so that 0 is white, black negative
    img = tf.math.scalar_mul(tf.constant(0.3, dtype=tf.dtypes.float32), img) # scale towards 0 to make it less
    img = img + 255 # shift so that 0 is white, black negative
    return img


def batch_bleed_through(img):
    """
    Randomly selects another line within the batch, flips it and puts a lighter version
    of it in the background. This is done to simulate text bleeding through from the other
    side of the page.

    :param img: a batch of lines to augment.
    :return: the batch of lines simulating text bleeding through the page.
    """
    rolled = tf.roll(img, shift=tf.random.uniform(shape=[], maxval=tf.shape(img)[0], dtype=tf.int32), axis=0) # roll on batch dimensions
    return tf.minimum(img, reverse_and_lighten(rolled))


def double_batch_bleed_through(img):
    """
    Randomly selects two other lines within the batch, flips thems and puts a lighter version
    of them in the background. The lines are shifted up a random amount so that the main line
    falls between them. This is done to simulate text bleeding through from the other
    side of the page.
    """
    rolled = tf.roll(img, shift=tf.random.uniform(shape=[], maxval=tf.shape(img)[0], dtype=tf.int32), axis=0) # roll on batch dimensions
    rolled2 = tf.roll(img, shift=tf.random.uniform(shape=[], maxval=tf.shape(img)[0], dtype=tf.int32), axis=0) # roll on batch dimensions
    rolled = tf.concat([rolled, rolled2], 1)
    rolled = tf.slice(rolled, [0, tf.random.uniform(shape=[], maxval=tf.shape(rolled)[1] - tf.shape(img)[1], dtype=tf.int32), 0, 0], tf.shape(img))
    return tf.minimum(img, reverse_and_lighten(rolled))
