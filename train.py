import sys

import matplotlib.pyplot as plt
import yaml
import tensorflow as tf

import hwr.dataset as ds
from hwr.training import ModelTrainer
from hwr.training import ModelMetrics

TRAIN_CSV_PATH = 'train_csv_path'
VAL_CSV_PATH = 'val_csv_path'
SPLIT_TRAIN = 'split_train'
TRAIN_SIZE = 'train_size'
MODEL_OUT = 'model_out'
MODEL_IN = 'model_in'
EPOCHS = 'epochs'
BATCH_SIZE = 'batch_size'
LEARNING_RATE = 'learning_rate'
MAX_SEQ_SIZE = 'max_seq_size'
IMG_SIZE = 'img_size'
SHOW_GRAPHS = 'show_graphs'
INCLUDE_METRICS = 'include_metrics'
SAVE_EVERY = 'save_every'


def train_model(args):
    """
    Train the model according to the images and labels specified

    python train.py <TRAIN_CONFIG_FILE>

    Command Line Arguments:
    * TRAIN_CONFIG_FILE (required): The path to the train configuration file. A train configuration file
      is provided as "train_config.yaml".

    Configuration File Arguments:
    * csv_path: The path to a tab-delimited CSV file containing | IMG_PATH | TRANSCRIPTION |
                * Note that the IMG_PATH is relative to the location of the CSV
    * model_out: The path to store the trained model weights after training
    * model_in: The path to pre-trained model weights to be loaded before training begins
    * epochs: The number of epochs to train
    * batch_size: The number of images in a mini-batch
    * learning_rate: The learning rate the optimizer uses during training
    * max_seq_size: The max number of characters in a line-level transcription
    * img_size: The size which all images will be resized for training
    * split_train: Whether or not to split the training set into a train/validation using the train_size parameter.
                   Train = train_size, Val = (1 - train_size)
    * train_size: The ratio used to determine the size of the train/validation split.
                  Used ONLY if split_train is set to True
    * show_graphs: Whether or not to show graphs of the loss after training
    * metrics: Whether or not to include metrics other than loss on the validation set

    :param args: command line arguments
    """
    # Ensure the train config file is included
    if len(args) == 0:
        print('Must include path to train config file. The default file is included as "train_config.yaml.')
        return

    # Read arguments from the config file:
    with open(args[0]) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    # Print available devices so we know if we are using CPU or GPU
    tf.print('Devices Available:', tf.config.list_physical_devices())

    char2idx = ds.get_char2idx()
    idx2char = ds.get_idx2char()

    # Create train/validation datasets depending on configuration settings
    # Split the train dataset based on the TRAIN_SIZE parameter
    if configs[SPLIT_TRAIN]:
        dataset_size = 100  # ds.get_dataset_size(configs[TRAIN_CSV_PATH])
        train_dataset_size = int(configs[TRAIN_SIZE] * dataset_size)
        val_dataset_size = dataset_size - train_dataset_size

        dataset = ds.get_encoded_dataset_from_csv(configs[TRAIN_CSV_PATH], char2idx, configs[MAX_SEQ_SIZE],
                                                  eval(configs[IMG_SIZE])).take(100)
        train_dataset = dataset.take(train_dataset_size)\
                               .shuffle(100, reshuffle_each_iteration=True)\
                               .batch(configs[BATCH_SIZE])
        val_dataset = dataset.skip(train_dataset_size)\
                             .batch(configs[BATCH_SIZE])

    else:  # Use the data as given in the train/validation csv files - no additional splits performed
        train_dataset_size = ds.get_dataset_size(configs[TRAIN_CSV_PATH])
        val_dataset_size = ds.get_dataset_size(configs[VAL_CSV_PATH])

        train_dataset = ds.get_encoded_dataset_from_csv(configs[TRAIN_CSV_PATH], char2idx, configs[MAX_SEQ_SIZE],
                                                        eval(configs[IMG_SIZE]))\
                          .shuffle(100, reshuffle_each_iteration=True)\
                          .batch(configs[BATCH_SIZE])
        val_dataset = ds.get_encoded_dataset_from_csv(configs[VAL_CSV_PATH], char2idx, configs[MAX_SEQ_SIZE],
                                                      eval(configs[IMG_SIZE])).batch(configs[BATCH_SIZE])

    # Train the model
    model_trainer = ModelTrainer(configs[EPOCHS], configs[BATCH_SIZE], train_dataset, train_dataset_size, val_dataset,
                                 val_dataset_size, configs[MODEL_OUT], model_in=configs[MODEL_IN], lr=configs[LEARNING_RATE],
                                 max_seq_size=configs[MAX_SEQ_SIZE], save_every=configs[SAVE_EVERY])
    model, losses = model_trainer.train()

    # Print the losses over the course of training
    print('Train Losses:', losses[0])
    print('Val Losses:', losses[1])

    if configs[INCLUDE_METRICS]:
        model_metrics = ModelMetrics(model, val_dataset, idx2char)
        cer, wer = model_metrics.get_error_rates()
        print('Character Error Rate: {:.2f}%'.format(cer * 100))
        print('Word Error Rate: {:.2f}%'.format(wer * 100))

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
