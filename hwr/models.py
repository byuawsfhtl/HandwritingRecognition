import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.constraints as kc
import numpy as np

from hwr.layers import FullGatedConv2D
from hwr.layers import GateBlock


class FlorRecognizer(tf.keras.Model):
    """
    Handwriting Recognition Model as described in the Blog Post,
    https://medium.com/@arthurflor23/handwritten-text-recognition-using-tensorflow-2-0-f4352b7afe16.
    This model combines ideas from the following papers:
    - https://ieeexplore-ieee-org.erl.lib.byu.edu/document/8270042
    - http://www.jpuigcerver.net/pubs/jpuigcerver_icdar2017.pdf
    """
    def __init__(self, vocabulary_size=197):
        """
        Define the model in terms of Keras layers

        :param vocabulary_size: The number of possible classes that a character could belong to
        """
        super(FlorRecognizer, self).__init__(name='flor_recognizer')

        self.permute = kl.Permute([2, 1, 3])
        self.conv1 = tf.keras.Sequential(name='conv1')
        self.conv1.add(
            kl.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer="he_uniform"))
        self.conv1.add(kl.PReLU(shared_axes=[1, 2]))
        self.conv1.add(kl.BatchNormalization(renorm=True))
        self.conv1.add(FullGatedConv2D(filters=16, kernel_size=(3, 3), padding="same"))

        self.conv2 = tf.keras.Sequential(name='conv2')
        self.conv2.add(
            kl.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform"))
        self.conv2.add(kl.PReLU(shared_axes=[1, 2]))
        self.conv2.add(kl.BatchNormalization(renorm=True))
        self.conv2.add(FullGatedConv2D(filters=32, kernel_size=(3, 3), padding="same"))

        self.conv3 = tf.keras.Sequential(name='conv3')
        self.conv3.add(
            kl.Conv2D(filters=64, kernel_size=(2, 4), strides=(2, 4), padding="same", kernel_initializer="he_uniform"))
        self.conv3.add(kl.PReLU(shared_axes=[1, 2]))
        self.conv3.add(kl.BatchNormalization(renorm=True))
        self.conv3.add(
            FullGatedConv2D(filters=64, kernel_size=(3, 3), padding="same", kernel_constraint=kc.MaxNorm(4, [0, 1, 2])))
        self.dropout1 = kl.Dropout(rate=0.2, name='dropout1')

        self.conv4 = tf.keras.Sequential(name='conv4')
        self.conv4.add(
            kl.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform"))
        self.conv4.add(kl.PReLU(shared_axes=[1, 2]))
        self.conv4.add(kl.BatchNormalization(renorm=True))
        self.conv4.add(
            FullGatedConv2D(filters=128, kernel_size=(3, 3), padding="same", kernel_constraint=kc.MaxNorm(4, [0, 1, 2])))
        self.dropout2 = kl.Dropout(rate=0.2, name='dropout2')

        self.conv5 = tf.keras.Sequential(name='conv5')
        self.conv5.add(
            kl.Conv2D(filters=256, kernel_size=(2, 4), strides=(2, 4), padding="same", kernel_initializer="he_uniform"))
        self.conv5.add(kl.PReLU(shared_axes=[1, 2]))
        self.conv5.add(kl.BatchNormalization(renorm=True))
        self.conv5.add(
            FullGatedConv2D(filters=256, kernel_size=(3, 3), padding="same", kernel_constraint=kc.MaxNorm(4, [0, 1, 2])))
        self.dropout3 = kl.Dropout(rate=0.2, name='dropout3')

        self.conv6 = tf.keras.Sequential(name='conv6')
        self.conv6.add(
            kl.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform"))
        self.conv6.add(kl.PReLU(shared_axes=[1, 2]))
        self.conv6.add(kl.BatchNormalization(renorm=True))

        self.mp = kl.MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding="valid", name='mp')

        self.gru1 = tf.keras.Sequential(name='gru1')
        self.gru1.add(kl.Bidirectional(kl.GRU(units=256, return_sequences=True, dropout=0.5)))
        self.gru1.add(kl.Dense(units=512))
        self.gru1.add(kl.PReLU())

        self.gru2 = tf.keras.Sequential(name='gru2')
        self.gru2.add(kl.Bidirectional(kl.GRU(units=256, return_sequences=True, dropout=0.5)))
        self.gru2.add(kl.Dense(units=vocabulary_size))

    def call(self, x, training=False, **kwargs):
        """
        Forward pass of the Recognizer. The training parameter is passed to certain layers
        in the model that act differently during training than they do during inference.

        :param x: The input to the recognizer (batch, height, width, channels)
        :param training: T/F - Is the model in training or inference mode?
        :param kwargs: Additional parameters
        :return:
        """
        out = self.permute(x)
        # CNN
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.dropout1(out, training=training)
        out = self.conv4(out)
        out = self.dropout2(out, training=training)
        out = self.conv5(out)
        out = self.dropout3(out, training=training)
        out = self.conv6(out)

        # MaxPool and Reshape
        out = self.mp(out)
        # out = tf.squeeze(out)
        out = tf.reshape(out, (-1, out.shape[1], out.shape[2] * out.shape[3]))

        # RNN
        out = self.gru1(out)
        out = self.gru2(out)

        return out


class GTRRecognizer(tf.keras.Model):
    def __init__(self, height, width, sequence_size=128, vocabulary_size=197, avg_pool_height=8, num_gateblocks='auto',
                 std_gateblock_filters=512, pooling_gateblock_filters=128, **kwargs):
        super(GTRRecognizer, self).__init__(**kwargs)

        self.ln1 = kl.LayerNormalization(trainable=False)
        self.conv1 = kl.Conv2D(16, kernel_size=(1, 1), padding='same')
        self.bn1 = kl.BatchNormalization(renorm=True)
        self.softmax1 = kl.Softmax()
        self.conv2 = kl.DepthwiseConv2D(kernel_size=(13, 13), padding='same')
        self.bn2 = kl.BatchNormalization(renorm=True)
        self.drop1 = kl.SpatialDropout2D(0.75)
        self.ln2 = kl.LayerNormalization(trainable=False)

        mp_heights = int(np.log2(height) - np.log2(avg_pool_height))
        mp_widths = int(np.log2(width) - np.log2(sequence_size))

        num_pooling_gateblocks = max(mp_heights, mp_widths)

        if num_gateblocks == 'auto':
            num_gateblocks = num_pooling_gateblocks

        if num_gateblocks < num_pooling_gateblocks:
            raise Exception("Recognizer requires a minimum of {} gateblocks with the specified configuration".format(
                num_pooling_gateblocks))

        self.gateblocks = tf.keras.Sequential()

        num_standard_gateblocks = num_gateblocks - num_pooling_gateblocks
        for index in range(num_standard_gateblocks):
            self.gateblocks.add(GateBlock(std_gateblock_filters))

        for index in range(num_pooling_gateblocks):
            height = 2 if mp_heights > 0 else 1
            width = 2 if mp_widths > 0 else 1
            mp_heights -= 1
            mp_widths -= 1
            self.gateblocks.add(GateBlock(pooling_gateblock_filters, pool_size=(height, width)))

        self.ln3 = kl.LayerNormalization(trainable=False)
        self.drop2 = kl.SpatialDropout2D(0.5)
        self.conv3 = kl.Conv2D(vocabulary_size, kernel_size=(1, 1), padding='same')
        self.bn3 = kl.BatchNormalization(renorm=True)
        self.elu1 = kl.ELU()
        self.ap1 = kl.AveragePooling2D(pool_size=(avg_pool_height, 1))
        self.ln4 = kl.LayerNormalization(trainable=False)

    def call(self, x, training=False, **kwargs):
        out = self.ln1(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.softmax1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.drop1(out)
        out = self.ln2(out)
        out = tf.concat((x, out), 3)

        out = self.gateblocks(out)
        out = self.ln3(out)
        out = self.drop2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.elu1(out)
        out = self.ap1(out)
        out = tf.squeeze(out, 1)
        out = self.ln4(out)

        return out
