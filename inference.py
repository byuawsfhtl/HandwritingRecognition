import sys
import matplotlib.pyplot as plt
import os
import pandas as pd

# Don't print any logs when booting up TensorFlow
# Comment out this line if you are running into issues running TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf # noqa - Suppress PyCharm Pep8 format warning
from model.model import Recognizer # noqa
from data.inference_sequence import InferenceSequence # noqa
from util.encoder import Encoder # noqa
from util.arguments import parse_arguments # noqa


def inference(args):
    """
    Perform inference on images specified by the user

    Command Line Arguments:
    python inference.py --img_path <IMG_PATH> --out_path <OUTPUT_PATH> --console
    * img_path (required): The path to images to be inferred
    * out_path (required if console not specified): The output path to the results of the inference
    * console (optional): Print inference results to the console and show images

    :param args: Command line arguments
    :return: None
    """
    # Place command line arguments in arg_dict
    arg_dict = parse_arguments(args)

    # Set up verbose logging if it was specified
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = arg_dict['log_level']

    # Create the required objects for inference
    sequence = InferenceSequence(arg_dict['img_path'])
    encoder = Encoder('./data/misc/char_set.json')
    model = Recognizer()

    # Load the pre-trained model weights
    model.load_weights('./data/model_weights/hwr_model/run1')

    # Keep track of all inferences in list of tuples
    inferences = []

    # Iterate through each of the images and perform inference
    for img, img_name in sequence:
        pred = model(img)
        best_path_pred = tf.argmax(pred, axis=2)
        str_pred = encoder.idxs_to_str_batch(best_path_pred)[0]

        if arg_dict['console']:
            print(img_name + ":", str_pred)
            plt.imshow(tf.transpose(tf.squeeze(img)), cmap='gray')
            plt.show()
        else:
            inferences.append((img_name, str_pred))

    # Write to output CSV if we aren't printing to the console
    if not arg_dict['console']:
        df = pd.DataFrame(data=inferences)
        df.to_csv(arg_dict['out_path'], sep='\t', index=False, header=False)
        print('Output written to', arg_dict['out_path'])


if __name__ == '__main__':
    inference(sys.argv[1:])  # Pass arguments besides the first (which is the python file)
