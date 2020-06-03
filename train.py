import sys
import matplotlib.pyplot as plt
import os

# Don't print any logs when booting up TensorFlow
# Comment out this line if you are running into issues running TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf # noqa - Suppress PyCharm Pep8 format warning
from data.sequence import TrainSequence # noqa
from util.arguments import parse_train_arguments # noqa
from data.tfrecord import create_tfrecord_from_sequence, read_tfrecord # noqa
from model.training import ModelTrainer # noqa


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


def train_model(args):
    """
    Train the model according to the images and labels specified

    python train.py --img_path <IMG_PATH> --label_path <CSV_LABEL_PATH> --show_graphs --log_level <LOG_LEVEL>
                    --model_out <WEIGHTS_OUT_PATH> --epochs <NUM_EPOCHS> --batch_size <BATCH_SIZE>
                    --learning_rate <LEARNING_RATE> --max_seq_size <MAX_SEQ_SIZE> --train_size <TRAIN_SET_SPLIT_SIZE>
                    --tfrecord_out <TF_RECORD_OUT_PATH>

    Command Line Arguments:
    * img_path (required): The path to the images in the dataset
    * label_path (required): The path to the label CSV (Format: Word | Transcription - Tab-Delimited, No-Header)
    * show_graphs (optional): Whether or not to show graphs of metrics after training (default: don't show graphs)
    * log-level (optional): TensorFlow log-level {0, 1, 2, 3} (default: 3)
    * model_out (optional): The path to store the model weights (default: ./data/model_weights/hwr_model/run1)
    * epochs (optional): The number of epochs to train (default: 100)
    * batch_size (optional): The number of images in a mini-batch (default: 100)
    * learning_rate (optional): The learning rate the optimizer uses during training (default: 4e-4)
    * max_seq_size (optional): The max number of characters in a line-level transcription (default: 128)
    * train_size (optional): The ratio used to determine the size of the train/validation sets (default: 0.8)
    * tfrecord_out (optional): The path to the created tfrecords file (default: './data/misc/data.tfrecords)
    * weights_path (optional): The path to pre-trained model weights (default: None)

    :param args: command line arguments
    """
    # Place command line arguments in arg_dict
    arg_dict = parse_train_arguments(args)

    # Set up verbose logging if it was specified
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = arg_dict['log_level']

    # We can import data into tensor format through a Keras Sequence
    sequence = TrainSequence(arg_dict['img_path'], arg_dict['label_path'])

    # Convert the data into TfRecord format which should be much faster
    create_tfrecord_from_sequence(sequence, arg_dict['tfrecord_out'])

    # Load a TensorFlow Dataset from the tfrecords file
    dataset = tf.data.TFRecordDataset(arg_dict['tfrecord_out'])\
        .map(read_tfrecord)\
        .shuffle(buffer_size=1000, reshuffle_each_iteration=True)

    # Split the dataset into a training and testing set
    dataset_size = len(sequence)
    train_dataset_size = int(arg_dict['train_size'] * dataset_size)
    val_dataset_size = dataset_size - train_dataset_size

    train_dataset = dataset.take(train_dataset_size).batch(arg_dict['batch_size'])
    val_dataset = dataset.skip(train_dataset_size).batch(arg_dict['batch_size'])

    # Create the train object and load in the configuration settings
    train = ModelTrainer(arg_dict['epochs'], arg_dict['batch_size'], train_dataset, train_dataset_size,
                         val_dataset, val_dataset_size, lr=arg_dict['learning_rate'],
                         max_seq_size=arg_dict['max_seq_size'], weights_path=arg_dict['weights_path'])

    # Train the model
    model, losses = train()

    # Save the model weights to the specified path
    model.save_weights(arg_dict['model_out'])

    # Print loss graph if command line argument specified it
    if True:
        show_loss_graph(losses[0], losses[1])


if __name__ == '__main__':
    train_model(sys.argv[1:])  # Pass arguments besides the first (which is the python file)
