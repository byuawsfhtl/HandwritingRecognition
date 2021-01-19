import tensorflow as tf
import numpy as np
from tqdm import tqdm


class ModelTrainer:
    """
    Responsible for training the model. Scope becomes an issue when dealing with @tf.function.
    It's easier to place all of the training code into an object so we don't run into issues.
    Once the object is created, the __call__ method will train and return the results and the trained model.
    """

    def __init__(self, model, epochs, batch_size, train_dataset, train_dataset_size, val_dataset, val_dataset_size,
                 model_out, lr=4e-4, max_seq_size=128):
        """
        Set up necessary variables that will be used during training, including the model, optimizer,
        encoder, and other metrics.

        :param model: The recognition model
        :param epochs: Number of epochs to train the model
        :param batch_size: How many images are in a mini-batch during training
        :param train_dataset: Train Dataset that is mapped and batched (see train method for context)
        :param train_dataset_size: The number of images in the training set
        :param val_dataset: Validation Dataset that is mapped and batched (see train method for context)
        :param val_dataset_size: The number of images in the validation set
        :param model_out: The path to the location where the weights will be stored during and after training
        :param lr: The learning rate of the model
        :param max_seq_size: The maximum length of a line-level transcription (See Encoder for context)
        """
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_dataset_size = train_dataset_size
        self.val_dataset_size = val_dataset_size
        self.model_out = model_out
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
        self.max_seq_size = max_seq_size
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')

    def save_model(self):
        tf.print('Saving Model Weights to', self.model_out)
        self.model.save_weights(self.model_out)

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
            # min_val_loss is the threshold where model weights will start to be saved when the validation loss drops
            # below this point. Then the model will only save the weights if the lowest validation loss is achieved.
            min_val_loss = 100

            # Main loop to go through each dataset for n epochs
            for epoch in range(self.epochs):
                # Reset our metrics for each epoch
                self.train_loss.reset_states()
                self.val_loss.reset_states()

                # Train Loop
                train_loop = tqdm(total=int(np.ceil(self.train_dataset_size / self.batch_size)), position=0, leave=True)
                for images, labels in self.train_dataset:
                    # Take a train step and update our progress bar
                    self.train_step(images, labels)
                    train_loop.set_description('Train - Epoch: {}, Loss: {:.4f}'
                                               .format(epoch, self.train_loss.result()))
                    train_loop.update(1)
                train_loop.close()

                # Validation Loop
                val_loop = tqdm(total=int(np.ceil(self.val_dataset_size / self.batch_size)), position=0, leave=True)
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
                if val_losses[-1] < min_val_loss:
                    self.save_model()
                    min_val_loss = val_losses[-1]

        except Exception as e:
            print("Error: {0}".format(e))
        finally:
            tf.print('Finished Training')
            return self.model, (train_losses, val_losses)

    def __call__(self):
        """
        Calls self.train() method which contains main training loop.

        :return: The model and train/val losses as tuple
        """
        return self.train()
