import sys

import matplotlib.pyplot as plt
import yaml

import hwr.dataset as ds
from hwr.models import FlorRecognizer, GTRRecognizer
from hwr.training import ModelTrainer

TRAIN_CSV_PATH = 'train_csv_path'
VAL_CSV_PATH = 'val_csv_path'
SPLIT_TRAIN_SIZE = 'split_train_size'
RECOGNITION_ARCHITECTURE = 'recognition_architecture'
STD_GATEBLOCK_FILTERS = 'std_gateblock_filters'
POOLING_GATEBLOCK_FILTERS = 'pooling_gateblock_filters'
NUM_GATEBLOCKS = 'num_gateblocks'
AVG_POOL_HEIGHT = 'avg_pool_height'
MODEL_OUT = 'model_out'
MODEL_IN = 'model_in'
EPOCHS = 'epochs'
BATCH_SIZE = 'batch_size'
LEARNING_RATE = 'learning_rate'
MAX_SEQ_SIZE = 'max_seq_size'
IMG_SIZE = 'img_size'
SHOW_GRAPHS = 'show_graphs'
CHARSET = 'charset'


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
    * model_out: The path to store the trained model weights after training
    * model_in: The path to pre-trained model weights to be loaded before training begins
    * epochs: The number of epochs to train
    * batch_size: The number of images in a mini-batch
    * learning_rate: The learning rate the optimizer uses during training
    * max_seq_size: The max number of characters in a line-level transcription
    * img_size: The size which all images will be resized for training
    * show_graphs: Whether or not to show graphs of the loss after training
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
        train_dataset = dataset.take(train_dataset_size)\
            .cache()\
            .shuffle(1000, reshuffle_each_iteration=True)\
            .batch(configs[BATCH_SIZE])
        val_dataset = dataset.skip(train_dataset_size)\
            .cache()\
            .batch(configs[BATCH_SIZE])

    else:  # Use the data as given in the train/validation csv files - no additional splits performed
        train_dataset_size = ds.get_dataset_size(configs[TRAIN_CSV_PATH])
        val_dataset_size = ds.get_dataset_size(configs[VAL_CSV_PATH])

        train_dataset = ds.get_encoded_dataset_from_csv(configs[TRAIN_CSV_PATH], char2idx, configs[MAX_SEQ_SIZE],
                                                        eval(configs[IMG_SIZE]))\
            .cache()\
            .shuffle(100, reshuffle_each_iteration=True)\
            .batch(configs[BATCH_SIZE])
        val_dataset = ds.get_encoded_dataset_from_csv(configs[VAL_CSV_PATH], char2idx, configs[MAX_SEQ_SIZE],
                                                      eval(configs[IMG_SIZE]))\
            .cache()\
            .batch(configs[BATCH_SIZE])

    if configs[RECOGNITION_ARCHITECTURE] == 'gtr':
        model = GTRRecognizer(eval(configs[IMG_SIZE])[0], eval(configs[IMG_SIZE])[1],
                              sequence_size=configs[MAX_SEQ_SIZE],
                              vocabulary_size=len(charset) + 1, std_gateblock_filters=configs[STD_GATEBLOCK_FILTERS],
                              pooling_gateblock_filters=configs[POOLING_GATEBLOCK_FILTERS],
                              num_gateblocks=configs[NUM_GATEBLOCKS], avg_pool_height=configs[AVG_POOL_HEIGHT])
    elif configs[RECOGNITION_ARCHITECTURE] == 'flor':
        model = FlorRecognizer(vocabulary_size=len(charset) + 1)
    else:
        raise Exception('Unsupported recognition architecture: {}. Please choose a supported architecture: {}.'.format(
            configs[RECOGNITION_ARCHITECTURE], '["flor", "gtr"]'))

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

    # Print loss graph if command line argument specified it
    if configs[SHOW_GRAPHS]:
        show_loss_graph(losses[0], losses[1])


def show_loss_graph(train_losses, val_losses):
    """
    Creates a graph showing the loss curve over time.

    :param train_losses: list of the losses on the training set per epoch
    :param val_losses: list of the losses on the validation set per epoch
    """
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    train_model(sys.argv[1:])  # Pass arguments besides the first (which is the python file)
