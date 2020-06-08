import sys
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm

# Don't print any logs when booting up TensorFlow
# Comment out this line if you are running into issues running TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf # noqa - Suppress PyCharm Pep8 format warning
from src.hwr.model import Recognizer # noqa
from src.hwr.dataset.sequence import InferenceSequence # noqa
from src.hwr.util import Encoder # noqa
from src.hwr.util import parse_inference_arguments # noqa


def inference(args):
    """
    Perform inference on images specified by the user

    python inference.py --img_path <IMG_PATH> --out_path <OUTPUT_PATH> --console

    Command Line Arguments:
    * img_path (required): The path to images to be inferred
    * out_path (required if console not specified): The output path to the results of the inference
    * weights_path (required): The path to the pre-trained model weights
    * console (optional): Print inference results to the console and show images (default: None)
    * log_level (optional): TensorFlow log-level {0, 1, 2, 3} (default: 3)

    :param args: Command line arguments
    :return: None
    """
    # Place command line arguments in arg_dict
    arg_dict = parse_inference_arguments(args)

    # Set up verbose logging if it was specified
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = arg_dict['log_level']

    # Create the required objects for inference
    sequence = InferenceSequence(arg_dict['img_path'])
    encoder = Encoder()
    model = Recognizer()

    # Load the pre-trained model weights
    model.load_weights(arg_dict['weights_path'])

    # Keep track of all inferences in list of tuples
    inferences = []

    # Iterate through each of the images and perform inference
    for img, img_name in tqdm(sequence):
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
