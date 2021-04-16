import sys

import yaml

import hwr.dataset as ds
import hwr.augmentation as aug
from hwr.models import FlorRecognizer
from hwr.training import ModelTrainer
from hwr.util import remove_file_with_wildcard

TRAIN_CSV_PATH = 'train_csv_path'
VAL_CSV_PATH = 'val_csv_path'
SPLIT_TRAIN_SIZE = 'split_train_size'
MODEL_OUT = 'model_out'
MODEL_IN = 'model_in'
EPOCHS = 'epochs'
BATCH_SIZE = 'batch_size'
LEARNING_RATE = 'learning_rate'
MAX_SEQ_SIZE = 'max_seq_size'
IMG_SIZE = 'img_size'
CHARSET = 'charset'
APPLY_NOISE_AUGMENTATION = 'apply_noise_augmentation'
APPLY_BLEEDTHROUGH_AUGMENTATION = 'apply_bleedthrough_augmentation'
APPLY_GRID_WARP_AUGMENTATION = 'apply_grid_warp_augmentation'
GRID_WARP_INTERVAL = 'grid_warp_interval'
GRID_WARP_STDDEV = 'grid_warp_stddev'


def train_model(args):
    """
    Train the model according to the images and labels specified

    python train.py <TRAIN_CONFIG_FILE>

    Command Line Arguments:
    * TRAIN_CONFIG_FILE (required): The path to the train configuration file. A train configuration file
      is provided as "train_config.yaml".

    Configuration File Arguments:
    * train_csv_path: The path to a tab-delimited CSV file containing training data information formatted as:
                      | IMG_PATH | Transcription |
    * val_csv_path: The path to a tab_delimited CSV file containing validation data information formatted as:
                      | IMG_PATH | Transcription |     This field may be left blank if Split Train is set to True.
    * split_train_size: The ratio used to determine the size of the train/validation split. If split_train_size is set
                        to 0.8, then the training set will contain 80% of the data, and validation 20%. The dataset is
                        not shuffled before being split. If a val_csv_path is given, this parameter will not be used.
                        Otherwise, the training set will be split using this parameter.
    * apply_noise_augmentation: Whether or not to apply the noise augmentation to the training dataset
    * apply_bleedthrough_augmentation: Whether or not to apply the bleedthrough augmentation to the training dataset
    * apply_grid_warp_augmentation: Whether or not to apply the grid warp augmentation to the training dataset
    * grid_warp_interval: The interval in pixels between control points in the grid warp augmentation
    * grid_warp_stddev: The standard deviation required in the grid warp augmentation
    * model_out: The path to store the trained model weights after training
    * model_in: The path to pre-trained model weights to be loaded before training begins
    * epochs: The number of epochs to train
    * batch_size: The number of images in a mini-batch
    * learning_rate: The learning rate the optimizer uses during training
    * max_seq_size: The max number of characters in a line-level transcription
    * img_size: The size which all images will be resized for training
    * charset: String including all characters to be represented in the network (abcdef1234...)
               If no characters are specified, the default is used.

    :param args: command line arguments
    """
    # Ensure the train config file is included
    if len(args) == 0:
        print('Must include path to train config file. The default file is included as "train_config.yaml.')
        return

    # Read arguments from the config file:
    with open(args[0]) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    # Remove previous cache files if they currently exist
    remove_file_with_wildcard("train.cache.*")
    remove_file_with_wildcard("validation.cache.*")

    charset = configs[CHARSET] if configs[CHARSET] else ds.DEFAULT_CHARS  # If no charset is given, use default
    char2idx = ds.get_char2idx(charset=charset)

    # Create train/validation datasets depending on configuration settings

    # Split the train dataset depending on if the val_csv_path is empty
    if not configs[VAL_CSV_PATH]:  # Will evaluate to False if empty
        dataset_size = ds.get_dataset_size(configs[TRAIN_CSV_PATH])
        train_dataset_size = int(configs[SPLIT_TRAIN_SIZE] * dataset_size)
        val_dataset_size = dataset_size - train_dataset_size

        dataset = ds.get_encoded_dataset_from_csv(configs[TRAIN_CSV_PATH], char2idx, configs[MAX_SEQ_SIZE],
                                                  eval(configs[IMG_SIZE]))
        train_dataset = dataset.take(train_dataset_size).cache("train.cache")\
            .shuffle(100, reshuffle_each_iteration=True)\
            .batch(configs[BATCH_SIZE])
        val_dataset = dataset.skip(train_dataset_size)\
            .batch(configs[BATCH_SIZE]).cache("validation.cache")

    else:  # Use the data as given in the train/validation csv files - no additional splits performed
        train_dataset_size = ds.get_dataset_size(configs[TRAIN_CSV_PATH])
        val_dataset_size = ds.get_dataset_size(configs[VAL_CSV_PATH])

        train_dataset = ds.get_encoded_dataset_from_csv(configs[TRAIN_CSV_PATH], char2idx, configs[MAX_SEQ_SIZE],
                                                        eval(configs[IMG_SIZE]))\
            .cache("train.cache").shuffle(100, reshuffle_each_iteration=True)\
            .batch(configs[BATCH_SIZE])
        val_dataset = ds.get_encoded_dataset_from_csv(configs[VAL_CSV_PATH], char2idx, configs[MAX_SEQ_SIZE],
                                                      eval(configs[IMG_SIZE]))\
            .batch(configs[BATCH_SIZE]).cache("validation.cache")

    if configs[APPLY_GRID_WARP_AUGMENTATION] or configs[APPLY_BLEEDTHROUGH_AUGMENTATION] \
            or configs[APPLY_NOISE_AUGMENTATION]:
        train_dataset = aug.augment_batched_dataset(train_dataset, random_noise=configs[APPLY_NOISE_AUGMENTATION],
                                                    bleedthrough=configs[APPLY_BLEEDTHROUGH_AUGMENTATION],
                                                    random_grid_warp=configs[APPLY_GRID_WARP_AUGMENTATION],
                                                    rgw_interval=configs.get(GRID_WARP_INTERVAL, None),
                                                    rgw_stddev=configs.get(GRID_WARP_STDDEV, None))

    model = FlorRecognizer(vocabulary_size=len(charset) + 1)
    if configs[MODEL_IN]:
        model.load_weights(configs[MODEL_IN])

    # Train the model
    model_trainer = ModelTrainer(model, configs[EPOCHS], configs[BATCH_SIZE], train_dataset, train_dataset_size,
                                 val_dataset, val_dataset_size, configs[MODEL_OUT], lr=configs[LEARNING_RATE],
                                 max_seq_size=configs[MAX_SEQ_SIZE])
    model, losses = model_trainer.train()

    # Print the losses over the course of training
    print('Train Losses:', losses[0])
    print('Val Losses:', losses[1])


if __name__ == '__main__':
    train_model(sys.argv[1:])  # Pass arguments besides the first (which is the python file)
