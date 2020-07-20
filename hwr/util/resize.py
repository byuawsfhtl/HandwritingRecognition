import numpy as np


def resize_img(img, desired_size):
    """
    Function for resizing an image. Note that the image must be able to
    be converted to a numpy array.

    :param img: Image that can be converted to numpy array
    :param desired_size: Tuple representing the shape of the image after resize
    :return: The image resized
    """
    # This needs to be figured out.... If we are given an image that is vertical, do we rotate it?
    # Should this even be something that the Handwriting Recognizer handles?
    if img.size[1] > img.size[0]:
        img = img.rotate(angle=90, expand=True)

    current_size = np.array(img).shape

    img_ratio = current_size[0] / current_size[1]
    desired_ratio = desired_size[0] / desired_size[1]

    # Calculate new height/width while preserving dimensions
    if img_ratio >= desired_ratio:
        new_height = desired_size[0]
        new_width = int(desired_size[0] // img_ratio)
    else:
        new_height = int(desired_size[1] * img_ratio)
        new_width = desired_size[1]

    img = np.array(img.resize((new_width, new_height)))

    border_top = desired_size[0] - new_height
    border_right = desired_size[1] - new_width

    # Ensure image pixels are 0-255
    # There may be a function we can use to ensure this is the case...
    img = np.pad(img, [(border_top, 0), (0, border_right)], mode='constant', constant_values=255)

    return img
