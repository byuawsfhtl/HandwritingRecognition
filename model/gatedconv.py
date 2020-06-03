import tensorflow.keras.layers as kl


class FullGatedConv2D(kl.Conv2D):
    """
    FullGatedConv2D

    Gated Convolutional Layer as described in the paper, Gated Convolutional Recurrent Neural Networks for Multilingual
    Handwriting Recognition (https://ieeexplore-ieee-org.erl.lib.byu.edu/document/8270042). Code obtained from the Flor
    implementation ()
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
