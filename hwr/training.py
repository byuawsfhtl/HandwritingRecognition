import tensorflow as tf
import numpy as np
from tqdm import tqdm

from hwr.model import Recognizer
from hwr.metrics import ErrorRates
from hwr.util import model_inference
from hwr.wbs.decoder import WordBeamSearch

import hwr.dataset as ds


class ModelTrainer:
    """
    Train

    Responsible for training the model. Scope becomes an issue when dealing with @tf.function.
    It's easier to place all of the training code into an object so we don't run into issues.
    Once the object is created, the __call__ method will train and return the results and the trained model.
    """

    def __init__(self, epochs, batch_size, train_dataset, train_dataset_size, val_dataset, val_dataset_size,
                 model_out, model_in=None, lr=4e-4, max_seq_size=128, save_every=5):
        """
        Set up necessary variables that will be used during training, including the model, optimizer,
        encoder, and other metrics.

        :param epochs: Number of epochs to train the model
        :param batch_size: How many images are in a mini-batch during training
        :param train_dataset: Train Dataset that is mapped and batched (see train_model function for context)
        :param train_dataset_size: The number of images in the training set
        :param val_dataset: Validation Dataset that is mapped and batched (see train_model function for context)
        :param val_dataset_size: The number of images in the validation set
        :param model_out: The path to the location where the weights will be stored during and after training
        :param model_in: The path to the weights of a pre-trained model that will be applied to the model prior to train
        :param lr: The learning rate of the model
        :param max_seq_size: The maximum length of a line-level transcription (See Encoder for context)
        :param save_every: The frequency which the model weights will be saved during training
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_dataset_size = train_dataset_size
        self.val_dataset_size = val_dataset_size
        self.model_out = model_out
        self.save_every = save_every

        self.model = Recognizer()
        if model_in is not None:  # Load the model weights before training - useful for fine-tuning
            self.model.load_weights(model_in)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        self.max_seq_size = max_seq_size
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')

    @tf.function
    def train_step(self, images, labels):
        """
        One training step given a mini-batch of images and labels. Note the use of the annotation, @tf.function.
        This annotation will allow TensorFlow to analyze the method and speed up training. However you must be
        careful on what can go inside a @tf.function. See the following links for details:
        * https://www.tensorflow.org/api_docs/python/tf/function
        * https://www.tensorflow.org/guide/function
        * https://pgaleone.eu/tensorflow/tf.function/2019/03/21/dissecting-tf-function-part-1/
        * https://pgaleone.eu/tensorflow/tf.function/2019/04/03/dissecting-tf-function-part-2/

        :param images: mini-batch of images in tensor format
        :param labels: mini-batch of index labels in tensor format
        """
        with tf.GradientTape() as tape:
            iter_batch_size = images.shape[0]

            # Items needed for CTC-Loss
            input_lengths = tf.constant(np.full((iter_batch_size,), self.max_seq_size))
            label_lengths = tf.math.count_nonzero(labels, axis=1)
            unique_labels = tf.nn.ctc_unique_labels(labels)

            # Make a prediction based on a batch of images
            predictions = self.model(images, training=True)

            # Calculate the loss
            loss = tf.nn.ctc_loss(labels, predictions, label_lengths, input_lengths,
                                  logits_time_major=False, unique=unique_labels)
            loss = tf.reduce_mean(loss)

        # Gradient Tape caught the gradients, now apply the gradients to the model
        # using the optimizer and update our loss metric
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.train_loss(loss)

    @tf.function
    def validation_step(self, images, labels):
        """
        One validation step given a mini-batch of images and labels. Note the use of the annotation, @tf.function.
        This annotation will allow TensorFlow to analyze the method and speed up training. However you must be
        careful on what can go inside a @tf.function. See the following links for details:
        * https://www.tensorflow.org/api_docs/python/tf/function
        * https://www.tensorflow.org/guide/function
        * https://pgaleone.eu/tensorflow/tf.function/2019/03/21/dissecting-tf-function-part-1/
        * https://pgaleone.eu/tensorflow/tf.function/2019/04/03/dissecting-tf-function-part-2/

        :param images: mini-batch of images in tensor format
        :param labels: mini-batch of index labels in tensor format
        """
        iter_batch_size = images.shape[0]

        # Items needed for CTC-Loss
        input_lengths = tf.constant(np.full((iter_batch_size,), self.max_seq_size))
        label_lengths = tf.math.count_nonzero(labels, axis=1)
        unique_labels = tf.nn.ctc_unique_labels(labels)

        # Make a prediction based on a batch of images
        predictions = self.model(images)

        # Calculate the loss and update the metric
        loss = tf.nn.ctc_loss(labels, predictions, label_lengths, input_lengths,
                              logits_time_major=False, unique=unique_labels)
        loss = tf.reduce_mean(loss)
        self.val_loss(loss)

    def train(self):
        """
        Main training loop. The method will training for the specified number of epochs and
        will iterate through the training and validation sets each epoch. The training and
        validation loss will be printed to the screen with a progress bar.

        :return: The model and losses during training
        """
        train_losses, val_losses = [], []

        # Place in a try/except and return the model/metrics in case we want to stop midway through training
        try:
            # Main loop to go through each dataset for n epochs
            for epoch in range(self.epochs):
                # Reset our metrics for each epoch
                self.train_loss.reset_states()
                self.val_loss.reset_states()

                # Train Loop
                train_loop = tqdm(total=self.train_dataset_size // self.batch_size, position=0, leave=True)
                for images, labels in self.train_dataset:
                    # Take a train step and update our progress bar
                    self.train_step(images, labels)
                    train_loop.set_description('Train - Epoch: {}, Loss: {:.4f}'
                                               .format(epoch, self.train_loss.result()))
                    train_loop.update(1)
                train_loop.close()

                # Validation Loop
                val_loop = tqdm(total=self.val_dataset_size // self.batch_size, position=0, leave=True)
                for images, labels in self.val_dataset:
                    # Take a validation step and update our progress bar
                    self.validation_step(images, labels)
                    val_loop.set_description('Val   - Epoch: {}, Loss: {:.4f}'
                                             .format(epoch, self.val_loss.result()))
                    val_loop.update(1)
                val_loop.close()

                train_losses.append(self.train_loss.result().numpy())
                val_losses.append(self.val_loss.result().numpy())

                # Save the model weights to the specified path
                if epoch % self.save_every == self.save_every - 1:
                    print('Saving Model Weights to', self.model_out)
                    self.model.save_weights(self.model_out)

        except Exception as e:
            print("Error: {0}".format(e))
        finally:
            # Save the model weights one last time and return the model/losses
            print('Finished Training')
            print('Saving Model Weights to', self.model_out)
            self.model.save_weights(self.model_out)
            return self.model, (train_losses, val_losses)

    def __call__(self):
        """
        Calls self.train() method which contains main training loop.

        :return: The model and train/val losses as tuple
        """
        return self.train()


class ModelMetrics:
    """
    ModelMetrics

    Used to provide metrics for a trained model.
    """

    def __init__(self, model, dataset, idx2char, wbs_decode=False):
        """
        :param model: The trained model to be tested
        :param dataset: The dataset likely in tf.data.dataset or Keras Sequence format
        :param idx2char:
        :param wbs_decode:
        """
        self.model = model
        self.dataset = dataset
        self.idx2char = idx2char
        self.wbs_decode = wbs_decode

        if self.wbs_decode:
            self.decoder = WordBeamSearch(25, 'Words', 0.0,)

    def get_error_rates(self):
        all_inferences = []
        all_labels = []
        for images, labels in self.dataset:
            output = model_inference(self.model, images)
            predictions = tf.argmax(output, axis=2)  # Best Path

            str_predictions = ds.idxs_to_str_batch(predictions, self.idx2char, merge_repeated=True)
            str_labels = ds.idxs_to_str_batch(labels, self.idx2char, merge_repeated=False)

            # Ensure we are working with unicode strings rather than byte strings
            str_predictions = [s.decode('utf8') if type(s) == np.bytes_ or type(s) == bytes else s
                               for s in str_predictions.numpy()]
            str_labels = [s.decode('utf8') if type(s) == np.bytes_ or type(s) == bytes else s
                          for s in str_labels.numpy()]

            all_inferences.extend(str_predictions)
            all_labels.extend(str_labels)

        # Map all inferences/labels to update the error rates
        rates = ErrorRates()

        for y_true, y_pred in zip(all_labels, all_inferences):
            rates.update(y_true, y_pred)
            rates.update(y_true, y_pred)

        return rates.get_error_rates()
