import sys

import tensorflow as tf
import pandas as pd
import yaml
from tqdm import tqdm

from hwr.model import Recognizer
from hwr.util import model_inference
import hwr.dataset as ds

IMG_PATH = 'img_path'
OUT_PATH = 'out_path'
MODEL_IN = 'model_in'
IMG_SIZE = 'img_size'
BATCH_SIZE = 'batch_size'
CONSOLE_OUT = 'console_out'


def inference(args):
    """
    Perform inference on images specified by the user

    python inference.py <INFERENCE_CONFIG_FILE>

    Command Line Arguments:
    * INFERENCE_CONFIG_FILE (required): The path to the inference configuration file. An inference configuration
      file is provided as "inference_config.yaml".

    Configuration File Arguments:
    * img_path: The path to the images to be inferred
    * out_path: The output path to the results of the inference
    * model_in: The path to the pre-trained model weights to be used during inference
    * img_size: The size which all images will be resized/padded for inference on the model
    * batch_size: The batch size to be used when performing inference on the model (how many images inferred at once)

    :param args: Command line arguments
    :return: None
    """
    # Ensure the inference config file is included
    if len(args) == 0:
        print('Must include path to the inference config file. The default file is included as inference_config.yaml')
        return

    with open(args[0]) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    # Print available devices so we know if we are using CPU or GPU
    tf.print('Devices Available:', tf.config.list_physical_devices())

    # Create the model object and load the pre-trained model weights
    model = Recognizer()
    model.load_weights(configs[MODEL_IN])

    dataset = ds.get_encoded_inference_dataset_from_img_path(configs[IMG_PATH], eval(configs[IMG_SIZE]))\
                .batch(configs[BATCH_SIZE])

    # Get the standard character set mapping
    idx2char = ds.get_idx2char()

    # Keep track of all inferences in list of tuples
    inferences = []

    # Iterate through each of the images and perform inference
    inference_loop = tqdm(total=dataset.cardinality().numpy(), position=0, leave=True)
    for imgs, img_names in dataset:
        output = model_inference(model, imgs)  # Without softmax
        predictions = tf.argmax(output, 2)
        str_predictions = ds.idxs_to_str_batch(predictions, idx2char)

        for str_pred, img_name in zip(str_predictions.numpy(), img_names.numpy()):
            inferences.append([str(str_pred, 'utf-8'), str(img_name, 'utf-8')])  # byte-string -> utf-8 string

        inference_loop.update(1)
    inference_loop.close()

    # Write to output CSV if we aren't printing to the console
    df = pd.DataFrame(data=inferences)
    df.to_csv(configs[OUT_PATH], sep='\t', index=False, header=False)
    print('Output written to', configs[OUT_PATH])


if __name__ == '__main__':
    inference(sys.argv[1:])  # Pass arguments besides the first (which is the python file)
