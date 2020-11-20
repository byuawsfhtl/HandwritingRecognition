import tensorflow as tf
import tensorflow.keras.layers as kl
import numpy as np


class FullGatedConv2D(kl.Conv2D):
    """
    Gated Convolutional Layer as described in the paper, Gated Convolutional Recurrent Neural Networks for Multilingual
    Handwriting Recognition (https://ieeexplore-ieee-org.erl.lib.byu.edu/document/8270042). Code obtained from the Flor
    implementation (https://github.com/arthurflor23/handwritten-text-recognition)
    """

    def __init__(self, filters, **kwargs):
        """
        :param filters: The number of filters to be used for the convolution
        :param kwargs: Additional kwargs to be passed to Conv2D
        """
        super(FullGatedConv2D, self).__init__(filters=filters * 2, **kwargs)
        self.nb_filters = filters

    def call(self, inputs):
        """
        Forward pass of gated convolutional layer

        :param inputs: The input to the gated convolution as tensor
        :return: The output of the gated convolution as tensor
        """
        output = super(FullGatedConv2D, self).call(inputs)
        linear = kl.Activation("linear")(output[:, :, :, :self.nb_filters])
        sigmoid = kl.Activation("sigmoid")(output[:, :, :, self.nb_filters:])
        return kl.Multiply()([linear, sigmoid])

    def compute_output_shape(self, input_shape):
        output_shape = super(FullGatedConv2D, self).compute_output_shape(input_shape)
        return tuple(output_shape[:3]) + (self.nb_filters,)

    def get_config(self):
        config = super(FullGatedConv2D, self).get_config()
        config['nb_filters'] = self.nb_filters
        del config['filters']
        return config


class GateBlock(tf.keras.Model):
    """
    A major layer in the GTRRecognizer Architecture. See the following paper for details:
    Accurate, Data-Efficient, Unconstrained Text Recognition with Convolutional Neural Networks
    https://arxiv.org/abs/1812.11894
    """
    def __init__(self, filters, pool_size=None):
        """
        Set up necessary layers for GateBlock

        :param filters: The number of output filters
        :param pool_size: The pool size used on the final max pooling layer. This parameter is sent directly to
                          tf.keras.MaxPooling2D, which requires int or tuple dtype. If none, no max pool layer is used.
        """
        super(GateBlock, self).__init__()

        filters_half = int(np.ceil(filters / 2))

        self.shortcut = kl.Conv2D(filters, kernel_size=(1, 1), use_bias=False)

        self.conv1 = kl.Conv2D(filters_half, kernel_size=(1, 1), padding='same')
        self.bn1 = kl.BatchNormalization(renorm=True)
        self.conv2 = kl.DepthwiseConv2D(kernel_size=(3, 3), padding='same')
        self.bn2 = kl.BatchNormalization(renorm=True)
        self.elu1 = kl.ELU()
        self.ln1 = kl.LayerNormalization(trainable=False)
        self.conv3 = kl.Conv2D(3 * filters_half, kernel_size=(1, 1), padding='same')
        self.bn3 = kl.BatchNormalization(renorm=True)
        self.conv4 = kl.DepthwiseConv2D(kernel_size=(3, 3), padding='same')
        self.bn4 = kl.BatchNormalization(renorm=True)
        self.ln2 = kl.LayerNormalization(trainable=False)
        self.ln3 = kl.LayerNormalization(trainable=False)
        self.ln4 = kl.LayerNormalization(trainable=False)
        self.ln5 = kl.LayerNormalization(trainable=False)
        self.conv5 = kl.Conv2D(filters, kernel_size=(1, 1), padding='same')
        self.bn5 = kl.BatchNormalization(renorm=True)
        self.conv6 = kl.DepthwiseConv2D(kernel_size=(3, 3), padding='same')
        self.bn6 = kl.BatchNormalization(renorm=True)
        self.elu2 = kl.ELU()

        if pool_size is not None:
            self.max_pool = True
            self.mp1 = kl.MaxPooling2D(pool_size=pool_size)
        else:
            self.max_pool = False

    def call(self, x, training=False, **kwargs):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.elu1(out)
        out = self.ln1(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.conv4(out)
        out = self.bn4(out)

        out1, out2, out3 = tf.split(out, 3, 3)  # Split into 3 even slices
        out1 = tf.keras.activations.tanh(out1)
        out2 = tf.keras.activations.tanh(out2)
        out3 = tf.keras.activations.sigmoid(out3)
        out1 = self.ln2(out1)
        out2 = self.ln3(out2)
        out3 = self.ln4(out3)
        out = out1 - out2
        out = out * out3

        out = self.ln5(out)
        out = self.conv5(out)
        out = self.bn5(out)
        out = self.conv6(out)
        out = self.bn6(out)
        out = self.elu2(out)

        # Apply shortcut if number of filters are different from input
        if tf.shape(x)[3] != tf.shape(out)[3]:
            x = self.shortcut(x)

        # Apply residual connection
        out = out + x

        if self.max_pool:
            out = self.mp1(out)

        return out
