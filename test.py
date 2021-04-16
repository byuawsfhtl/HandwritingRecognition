import sys

import numpy as np
import yaml
from tqdm import tqdm

import hwr.dataset as ds
from hwr.metrics import ErrorRates
from hwr.models import FlorRecognizer
from hwr.util import model_inference, bp_decode
from hwr.wbs.loader import DictionaryLoader
from hwr.wbs.decoder import WordBeamSearch


CSV_PATH = 'csv_path'
DATASET_EVAL_SIZE = 'dataset_eval_size'
MODEL_IN = 'model_in'
BATCH_SIZE = 'batch_size'
MAX_SEQ_SIZE = 'max_seq_size'
IMG_SIZE = 'img_size'
CHARSET = 'charset'
SHOW_PREDICTIONS = 'show_predictions'
WBS_BEAM_WIDTH = 'wbs_beam_width'
WBS_WORD_CHARSET = 'wbs_word_charset'
WBS_DICTIONARY = 'wbs_dictionary_path'


def test(args):
    """
    Test

    Usage:
    * python test.py <TEST_CONFIG_FILE>

    Command Line Arguments:
    * TEST_CONFIG_FILE: The path to the test configuration file. A test configuration file
      is provided as "test_config.yaml".

    Configuration File Arguments:
    * csv_path: The path to the csv file
    * dataset_eval_size: How much of the dataset should be used when testing and acquiring error rates.
                         Float between 0-1.
    * batch_size: The number of images to be used in a batch
    * max_seq_size: The max number of characters in a line-level transcription
    * charset: String including all characters to be represented in the network (abcdef1234...)
               If no characters are specified, the default is used.
    * show_predictions: Boolean indicating whether or not to print the bp/wbs predictions along with label
    * wbs_beam_width: The beam width needed for the word beam search algorithm
    * wbs_word_charset: String containing all word_charset characters
    * wbs_dictionary_path: A path to a file containing a list of words that the wbs should constrain to
                           If none given wbs will use the default dictionary
    """
    # Ensure the train config file is included
    if len(args) == 0:
        print('Must include path to test config file. The default file is included as "test_config.yaml.')
        return

    # Read arguments from the config file:
    with open(args[0]) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    charset = configs[CHARSET] if configs[CHARSET] else ds.DEFAULT_CHARS  # If no charset is given, use default
    word_charset = configs[WBS_WORD_CHARSET] if configs[WBS_WORD_CHARSET] else ds.DEFAULT_NON_PUNCTUATION
    char2idx = ds.get_char2idx(charset=charset)
    idx2char = ds.get_idx2char(charset=charset)

    dataset_size = ds.get_dataset_size(configs[CSV_PATH])
    dataset = ds.get_encoded_dataset_from_csv(configs[CSV_PATH], char2idx, configs[MAX_SEQ_SIZE],
                                              eval(configs[IMG_SIZE]))\
        .skip(int(dataset_size * (1 - configs[DATASET_EVAL_SIZE])))\
        .batch(configs[BATCH_SIZE])
    # Recalculate dataset size after skipping part of the dataset as specified by dataset_eval_size parameter
    dataset_size = dataset_size - int(dataset_size * (1 - configs[DATASET_EVAL_SIZE]))

    model = FlorRecognizer(vocabulary_size=len(charset) + 1)
    if configs[MODEL_IN]:
        model.load_weights(configs[MODEL_IN])

    # Corpus creation. Currently, this is a manual process of loading in specific dictionaries.
    if configs[WBS_DICTIONARY]:
        corpus = DictionaryLoader.from_file(configs[WBS_DICTIONARY], include_cased=True)
    else:
        corpus = DictionaryLoader.ascii_names(include_cased=True) + '\n' +\
            DictionaryLoader.english_words(include_cased=True)

    # Create the word beam search decoder
    wbs = WordBeamSearch(corpus, charset, word_charset, configs[WBS_BEAM_WIDTH])

    # Create lists to store labels and predictions for various decoding methods
    bp_predictions = []
    wbs_predictions = []
    actual_labels = []

    # Main inference loop iterating over the test dataset
    loop = tqdm(total=int(dataset_size/configs[BATCH_SIZE]), position=0, leave=True)
    for images, labels in dataset:
        # Run inference on the model
        output = model_inference(model, images)

        bp_prediction = bp_decode(output)
        wbs_prediction = wbs.decode(output)

        # Perform best-path decoding, map to strings, and append to prediction list
        str_bp_prediction = ds.idxs_to_str_batch(bp_prediction, idx2char, merge_repeated=True)
        bp_predictions.extend(bytes_to_unicode(str_bp_prediction))

        # Perform word-beam-search decoding, map to strings, and append to prediction list
        str_wbs_prediction = ds.idxs_to_str_batch(wbs_prediction, idx2char, merge_repeated=True)
        wbs_predictions.extend(bytes_to_unicode(str_wbs_prediction))

        # Get labels, map to strings, and append to label list
        str_labels = ds.idxs_to_str_batch(labels, idx2char, merge_repeated=False)
        actual_labels.extend(bytes_to_unicode(str_labels))
        loop.update(1)
    loop.close()

    # Calculate error rates by iterating over the predictions/labels
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

    # Print the error rates to the console
    print('Best Path - Character Error Rate: {:.4f}%'.format(bp_cer * 100))
    print('Best Path - Word Error Rate: {:.4f}%'.format(bp_wer * 100))
    print('Word Beam Search - Character Error Rate: {:.4f}%'.format(wbs_cer * 100))
    print('Word Beam Search - Word Error Rate: {:.4f}%'.format(wbs_wer * 100))


def bytes_to_unicode(byte_string_tensor):
    """
    Takes a tensor of byte strings and converts them to a regular python list of unicode strings.
    This is useful when string tensors need to be converted for error rate calculations.

    :param: byte_string_tensor: A tensor of byte strings
    :return: A python list of strings
    """
    return [s.decode('utf8') if type(s) == np.bytes_ or type(s) == bytes else s
            for s in byte_string_tensor.numpy()]


if __name__ == '__main__':
    test(sys.argv[1:])
