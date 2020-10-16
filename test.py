import sys

import tensorflow as tf
import numpy as np
import yaml
from tqdm import tqdm

import hwr.dataset as ds
from hwr.model import Recognizer
from hwr.metrics import ErrorRates
from hwr.util import model_inference
from hwr.wbs.loader import DictionaryLoader
from hwr.wbs.decoder import WordBeamSearch


CSV_PATH = 'csv_path'
MODEL_IN = 'model_in'
BATCH_SIZE = 'batch_size'
MAX_SEQ_SIZE = 'max_seq_size'
IMG_SIZE = 'img_size'
CHARSET = 'charset'
SHOW_PREDICTIONS = 'show_predictions'
WBS_BEAM_WIDTH = 'wbs_beam_width'
WBS_OS_TYPE = 'wbs_os_type'
WBS_GPU = 'wbs_gpu'
WBS_MULTITHREADED = 'wbs_multithreaded'


def test(args):
    """
    Test

    Usage:
    * python test.py <TEST_CONFIG_FILE>

    Command Line Arguments:
    * TEST_CONFIG_FILE: The path to the test configuration file. A test configuration file
      is provided as "test_config.yaml".

    Command Line Arguments:
    * csv_path: The path to the csv file
    * batch_size: The number of images to be used in a batch
    * charset: String including all characters to be represented in the network (abcdef1234...)
               If no characters are specified, the default is used.
    """
    # Ensure the train config file is included
    if len(args) == 0:
        print('Must include path to train config file. The default file is included as "train_config.yaml.')
        return

    # Read arguments from the config file:
    with open(args[0]) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    charset = configs[CHARSET] if not str(configs[CHARSET]) else None  # If no charset is given pass None to use default
    char2idx = ds.get_char2idx(charset=charset)
    idx2char = ds.get_idx2char(charset=charset)

    dataset = ds.get_encoded_dataset_from_csv(configs[CSV_PATH], char2idx, configs[MAX_SEQ_SIZE],
                                              eval(configs[IMG_SIZE]))\
        .batch(configs[BATCH_SIZE])
    dataset_size = ds.get_dataset_size(configs[CSV_PATH])

    model = Recognizer()  # Plus the ctc-blank character
    model.load_weights(configs[MODEL_IN])

    wbs = WordBeamSearch(configs[WBS_BEAM_WIDTH], 'Words', 0.0, DictionaryLoader.census_names_15(include_cased=True),
                         ds.DEFAULT_CHARS, ds.DEFAULT_NON_PUNCTUATION, os_type=configs[WBS_OS_TYPE],
                         gpu=configs[WBS_GPU], multithreaded=configs[WBS_MULTITHREADED])

    bp_predictions = []
    wbs_predictions = []
    actual_labels = []

    loop = tqdm(total=int(np.round(dataset_size/configs[BATCH_SIZE])), position=0, leave=True)
    for images, labels in dataset:
        # Run inference on the model
        output = model_inference(model, images)

        # Perform best-path decoding, map to strings, and append to prediction list
        bp_prediction = tf.argmax(output, axis=2)
        str_bp_prediction = ds.idxs_to_str_batch(bp_prediction, idx2char, merge_repeated=True)
        bp_predictions.extend(bytes_to_unicode(str_bp_prediction))

        # Perform word-beam-search decoding, map to strings, and append to prediction list
        wbs_prediction = wbs(output)
        str_wbs_prediction = ds.idxs_to_str_batch(wbs_prediction, idx2char, merge_repeated=False)
        wbs_predictions.extend(bytes_to_unicode(str_wbs_prediction))

        # Get labels, map to strings, and append to label list
        str_labels = ds.idxs_to_str_batch(labels, idx2char, merge_repeated=False)
        actual_labels.extend(bytes_to_unicode(str_labels))
        loop.update(1)

    bp_rates = ErrorRates()
    wbs_rates = ErrorRates()
    for bp_pred, wbs_pred, y_true in zip(bp_predictions, wbs_predictions, actual_labels):
        if configs[SHOW_PREDICTIONS]:
            print('Best Path Prediction:', bp_pred)
            print('Word Beam Search Prediction:', wbs_pred)
            print('Ground Truth:', y_true)
        bp_rates.update(y_true, bp_pred)
        wbs_rates.update(y_true, wbs_pred)

    bp_cer, bp_wer = bp_rates.get_error_rates()
    wbs_cer, wbs_wer = wbs_rates.get_error_rates()

    print('Best Path - Character Error Rate: {:.4f}%'.format(bp_cer * 100))
    print('Best Path - Word Error Rate: {:.4f}%'.format(bp_wer * 100))
    print('Word Beam Search - Character Error Rate: {:.4f}%'.format(wbs_cer * 100))
    print('Word Beam Search - Word Error Rate: {:.4f}%'.format(wbs_wer * 100))


def bytes_to_unicode(byte_string_tensor):
    """
    Takes a tensor of byte strings and converts them to a regular python list of unicode strings
    """
    return [s.decode('utf8') if type(s) == np.bytes_ or type(s) == bytes else s
            for s in byte_string_tensor.numpy()]


if __name__ == '__main__':
    test(sys.argv[1:])
