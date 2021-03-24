import sys

import tensorflow as tf
import pandas as pd
import yaml
from tqdm import tqdm

import hwr.dataset as ds
from hwr.models import FlorRecognizer
from hwr.util import model_inference, bp_decode
from hwr.wbs.loader import DictionaryLoader
from hwr.wbs.decoder import WordBeamSearch

IMG_PATH = 'img_path'
IMG_PATH_SUBDIRS = 'img_path_subdirs'
OUT_PATH = 'out_path'
MODEL_IN = 'model_in'
IMG_SIZE = 'img_size'
BATCH_SIZE = 'batch_size'
MAX_SEQ_SIZE = 'max_seq_size'
CHARSET = 'charset'

USE_WBS = 'use_wbs'
WBS_BEAM_WIDTH = 'wbs_beam_width'
WBS_WORD_CHARSET = 'wbs_word_charset'
WBS_DICTIONARY = 'wbs_dictionary_path'


def inference(args):
    """
    Perform inference on images specified by the user

    python inference.py <INFERENCE_CONFIG_FILE>

    Command Line Arguments:
    * INFERENCE_CONFIG_FILE (required): The path to the inference configuration file. An inference configuration
      file is provided as "inference_config.yaml".

    Configuration File Arguments:
    * img_path: The path to the images to be inferred
    * img_path_subdirs: Boolean whether or not to include subdirectories of img_path
    * out_path: The output path to the results of the inference
    * model_in: The path to the pre-trained model weights to be used during inference
    * img_size: The size which all images will be resized/padded for inference on the model
    * batch_size: The batch size to be used when performing inference on the model (how many images inferred at once)
    * max_seq_size: The max number of characters in a line-level transcription
    * charset: String including all characters to be represented in the network (abcdef1234...)
               If no characters are specified, the default is used.
    * use_wbs: Boolean indicating whether or not to use Word Beam Search for decoding. If False, best path is used.
    * wbs_beam_width: The beam width needed for the word beam search algorithm
    * wbs_word_charset: String containing all characters observed in words (non-word_charset)
    * wbs_dictionary_path: A path to a file containing a list of words that the wbs should constrain to
                           If none given wbs will use the default dictionary

    :param args: Command line arguments
    :return: None
    """
    # Ensure the inference config file is included
    if len(args) == 0:
        print('Must include path to the inference config file. The default file is included as inference_config.yaml')
        return

    with open(args[0]) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    # Create the tensorflow dataset with images to be inferred
    dataset = ds.get_encoded_inference_dataset_from_img_path(configs[IMG_PATH], eval(configs[IMG_SIZE]),
                                                             configs[IMG_PATH_SUBDIRS]).batch(configs[BATCH_SIZE])

    # Load our character set
    charset = configs[CHARSET] if configs[CHARSET] else ds.DEFAULT_CHARS  # If no charset is given, use default
    word_charset = configs[WBS_WORD_CHARSET] if configs[WBS_WORD_CHARSET] else ds.DEFAULT_NON_PUNCTUATION
    idx2char = ds.get_idx2char(charset=charset)

    model = FlorRecognizer(vocabulary_size=len(charset) + 1)
    if configs[MODEL_IN]:
        model.load_weights(configs[MODEL_IN])

    # Corpus creation. Currently, this is a manual process of loading in specific dictionaries.
    if configs[WBS_DICTIONARY]:
        corpus = DictionaryLoader.from_file(configs[WBS_DICTIONARY], include_cased=True)
    else:
        corpus = DictionaryLoader.ascii_names(include_cased=True) + '\n' +\
            DictionaryLoader.english_words(include_cased=True)

    if configs[USE_WBS]:
        wbs = WordBeamSearch(corpus, charset, word_charset, configs[WBS_BEAM_WIDTH])

    # Keep track of all inferences in list of tuples
    inferences = []

    # Iterate through each of the images and perform inference
    inference_loop = tqdm(total=tf.data.experimental.cardinality(dataset).numpy(), position=0, leave=True)
    for imgs, img_names in dataset:
        output = model_inference(model, imgs)
        if configs[USE_WBS]:  # Use Word Beam Search Decoding
            predictions = wbs.decode(output)
        else:  # Use Best Path Decoding
            predictions = bp_decode(output)

        # Convert predictions to strings
        str_predictions = ds.idxs_to_str_batch(predictions, idx2char, merge_repeated=True)

        # Append to inferences list
        for str_pred, img_name in zip(str_predictions.numpy(), img_names.numpy()):
            inferences.append([str(img_name, 'utf-8'), str(str_pred, 'utf-8')])  # byte-string -> utf-8 string

        inference_loop.update(1)
    inference_loop.close()

    # Write to output CSV if we aren't printing to the console
    df = pd.DataFrame(data=inferences)
    df.to_csv(configs[OUT_PATH], sep='\t', index=False, header=False)
    print('Output written to', configs[OUT_PATH])


if __name__ == '__main__':
    inference(sys.argv[1:])  # Pass arguments besides the first (which is the python file)
