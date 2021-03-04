import sys

import yaml
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button

from hwr.models import FlorRecognizer
from hwr.active import InstanceSelector
import hwr.dataset as ds

DATASET_PATH = 'dataset_path'
MODEL_IN = 'model_in'
MODEL_OUT = 'model_out'
IMG_SIZE = 'img_size'
CHARSET = 'charset'
NUM_ACTIVE_INSTANCES = 'num_active_instances'
ACTIVE_SAMPLE_SIZE = 'active_sample_size'

checked_status = False


def fit(args):
    # Ensure the train config file is included
    if len(args) == 0:
        print('Must include path to fit config file. The default file is included as "fit_config.yaml."')
        return

    # Read arguments from the config file:
    with open(args[0]) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    charset = configs[CHARSET] if configs[CHARSET] else ds.DEFAULT_CHARS  # If no charset is given, use default
    sample_size = configs[ACTIVE_SAMPLE_SIZE] if configs[ACTIVE_SAMPLE_SIZE] else -1

    model = FlorRecognizer(vocabulary_size=len(charset) + 1)
    model.load_weights(configs[MODEL_IN])

    selector = InstanceSelector(model, configs[DATASET_PATH], eval(configs[IMG_SIZE]), sample_size=sample_size)

    dataset = selector.select(configs[NUM_ACTIVE_INSTANCES])

    while dataset.cardinality().numpy() > 0:
        for img, img_name in dataset:
            plt.figure(figsize=(15, 5))
            plt.imshow(tf.squeeze(img), cmap='gray')

            axbox = plt.axes([0.1, 0.05, 0.8, 0.1])
            box = TextBox(axbox, 'Transcription:',)
            box.begin_typing(None)

            axbutton = plt.axes([0.85, 0.05, 0.1, 0.1])
            button = Button(axbutton, 'Submit')

            box.on_submit(lambda x: plt.close('all'))
            button.on_clicked(lambda x: plt.close('all'))

            plt.show()
            print('Given Text:', box.text)

            # transcription = input('Please provide transcription:')
            # print('Transcription:', transcription)
            # plt.close('all')

        break

        dataset = selector.select(configs[NUM_ACTIVE_INSTANCES])


if __name__ == '__main__':
    fit(sys.argv[1:])
