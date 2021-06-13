import tensorflow as tf
import tensorflow_addons as tfa


def augment_batched_dataset(ds, random_noise=True, bleedthrough=True, random_grid_warp=True,
                            rgw_interval=None, rgw_stddev=None):
    """
    Maps the batches within a dataset to augment them.

    :param ds: the dataset to augment
    :param random_noise: Boolean indicating whether or not to apply random noise to the background of the image
    :param bleedthrough: Boolean indicating whether or not to apply the bleedthrough augmentation
    :param random_grid_warp: Boolean indicating whether or not to apply the random grid warp augmentation
    :param rgw_interval: The interval in pixels between control points on the grid
    :param rgw_stddev: The standard deviation used when distorting control points by sampling from a normal distribution
    :return the augmented dataset
    """
    return ds.map(lambda img, trans: (augment_batch(img, noise_augmentation=random_noise,
                                                    bleedthrough_augmentation=bleedthrough,
                                                    random_grid_warp=random_grid_warp, rgw_interval=rgw_interval,
                                                    rgw_stddev=rgw_stddev), trans))


def augment_batch(batch, noise_augmentation=True, bleedthrough_augmentation=True, random_grid_warp=True,
                  rgw_interval=None, rgw_stddev=None):
    """
    Randomly applies different augmentations to the given batch.

    :param batch: the batch to augment
    :param noise_augmentation: Boolean indicating whether or not to apply random noise to the background of the image
    :param bleedthrough_augmentation: Boolean indicating whether or not to apply the bleedthrough augmentation
    :param random_grid_warp: Boolean indicating whether or not to apply the random grid warp augmentation
    :param rgw_interval: The interval in pixels between control points on the grid
    :param rgw_stddev: The standard deviation used when distorting control points by sampling from a normal distribution
    :return: the augmented batch
    """
    if bleedthrough_augmentation and tf.random.uniform([]) < 0.5:
        batch = double_batch_bleed_through(batch)
    if noise_augmentation and tf.random.uniform([]) < 0.5:
        batch = add_noise(batch)
    if random_grid_warp:
        batch = batch_random_grid_warp_distortions(batch, rgw_interval, rgw_stddev)

    return batch


def batch_random_grid_warp_distortions(batch, grid_interval, stddev):
    """
    Perform a Random Grid Warp Distortion Augmentation on a batch of images as given in the following paper:

    Data Augmentation for Recognition of Handwritten Words and Lines Using a CNN-LSTM Network
    https://ieeexplore.ieee.org/abstract/document/8270041

    @param batch: The batch of images (batch x height x width)
    @param grid_interval: The interval between control points on the grid
    @param stddev: The standard deviation used when distorting control points by sampling from a normal distribution

    @return: The warped batch
    """
    batch_shape = tf.shape(batch)

    # Create grid of control points based on height, width, and grid interval
    row_list = tf.range(0, batch_shape[1], delta=grid_interval, dtype=tf.float32)
    column_list = tf.range(0, batch_shape[2], delta=grid_interval, dtype=tf.float32)
    control_points = tf.transpose([tf.tile(row_list, tf.shape(column_list)), tf.repeat(column_list, tf.shape(row_list)[0])])

    # Sample from normal distribution with mean 0 and stddev as given by parameter
    random_distortions = tf.random.normal((batch_shape[0], tf.shape(control_points)[0], tf.shape(control_points)[1]),
                                          mean=0, stddev=stddev)

    # Apply the distortion - with broadcasting
    destination_points = control_points + random_distortions

    # Tensorflow add-ons uses non-tf functions to perform this operation
    warped_batch, _ = tfa.image.sparse_image_warp(batch, control_points, destination_points)

    return warped_batch


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
    img = tf.reverse(img, [2])  # reverse on x dimension
    img = img - 255  # shift so that 0 is white, black negative
    img = tf.math.scalar_mul(tf.constant(0.3, dtype=tf.dtypes.float32), img)  # scale towards 0 to make it less
    img = img + 255  # shift so that 0 is white, black negative
    return img


def batch_bleed_through(img):
    """
    Randomly selects another line within the batch, flips it and puts a lighter version
    of it in the background. This is done to simulate text bleeding through from the other
    side of the page.

    :param img: a batch of lines to augment.
    :return: the batch of lines simulating text bleeding through the page.
    """
    # roll on batch dimensions
    rolled = tf.roll(img, shift=tf.random.uniform(shape=[], maxval=tf.shape(img)[0], dtype=tf.int32), axis=0)
    return tf.minimum(img, reverse_and_lighten(rolled))


def double_batch_bleed_through(img):
    """
    Randomly selects two other lines within the batch, flips thems and puts a lighter version
    of them in the background. The lines are shifted up a random amount so that the main line
    falls between them. This is done to simulate text bleeding through from the other
    side of the page.
    """
    # roll on batch dimensions
    rolled = tf.roll(img, shift=tf.random.uniform(shape=[], maxval=tf.shape(img)[0], dtype=tf.int32), axis=0)
    rolled2 = tf.roll(img, shift=tf.random.uniform(shape=[], maxval=tf.shape(img)[0], dtype=tf.int32), axis=0)
    rolled = tf.concat([rolled, rolled2], 1)
    rolled = tf.slice(rolled, [0, tf.random.uniform(shape=[], maxval=tf.shape(rolled)[1] - tf.shape(img)[1],
                                                    dtype=tf.int32), 0, 0], tf.shape(img))
    return tf.minimum(img, reverse_and_lighten(rolled))
