import tensorflow as tf

class ResNet12Block(tf.keras.layers.Layer):
    def __init__(self, num_filters, add_relu=True):
        """
        ResNet12 Block from https://arxiv.org/abs/1805.10123
        Swapped swish activation for relu.
        :param num_filters:
        :param add_relu:
        """
        super(ResNet12Block, self).__init__()

        self._num_filters = num_filters
        self._add_relu = add_relu

    def build(self, input_shape):
        self._shortcut_conv = tf.keras.layers.Convolution2D(filters=self._num_filters, kernel_size=1, strides=1, padding="SAME")
        self._shortcut_bn = tf.keras.layers.BatchNormalization()

        self._conv1 = tf.keras.layers.Convolution2D(filters=self._num_filters, kernel_size=3, strides=1, padding="SAME")
        self._bn1 = tf.keras.layers.BatchNormalization()

        self._conv2 = tf.keras.layers.Convolution2D(filters=self._num_filters, kernel_size=3, strides=1, padding="SAME")
        self._bn2 = tf.keras.layers.BatchNormalization()

        self._conv3 = tf.keras.layers.Convolution2D(filters=self._num_filters, kernel_size=3, strides=1, padding="SAME")
        self._bn3 = tf.keras.layers.BatchNormalization()

        self._maxpool = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="SAME")

    def call(self, inputs, training=None, **kwargs):
        shortcut = self._shortcut_conv(inputs)
        shortcut = self._shortcut_bn(shortcut, training=training)

        x = self._conv1(inputs)
        x = self._bn1(x, training=training)
        x = tf.keras.activations.relu(x)

        x = self._conv2(x)
        x = self._bn2(x, training=training)
        x = tf.keras.activations.relu(x)

        x = self._conv3(x)
        x = self._bn3(x, training=training)
        x = tf.keras.activations.relu(x)

        x = x + shortcut
        x = self._maxpool(x)

        if self._add_relu:
            x = tf.keras.activations.relu(x)

        return x

class ResNet12BlockReverse(tf.keras.layers.Layer):
    def __init__(self, num_filters, add_relu=True):
        """
        Inverse function of ResNet12 block for decoding.
        :param num_filters:
        :param add_relu:
        """
        super(ResNet12BlockReverse, self).__init__()

        self._num_filters = num_filters
        self._add_relu = add_relu

    def build(self, input_shape):
        self._shortcut_conv = tf.keras.layers.Conv2DTranspose(filters=self._num_filters, kernel_size=1, strides=1, padding="SAME")
        self._shortcut_bn = tf.keras.layers.BatchNormalization()

        self._conv1 = tf.keras.layers.Conv2DTranspose(filters=self._num_filters, kernel_size=3, strides=1, padding="SAME")
        self._bn1 = tf.keras.layers.BatchNormalization()

        self._conv2 = tf.keras.layers.Conv2DTranspose(filters=self._num_filters, kernel_size=3, strides=1, padding="SAME")
        self._bn2 = tf.keras.layers.BatchNormalization()

        self._conv3 = tf.keras.layers.Conv2DTranspose(filters=self._num_filters, kernel_size=3, strides=1, padding="SAME")
        self._bn3 = tf.keras.layers.BatchNormalization()

        self._upsample = tf.keras.layers.UpSampling2D(size=2)

    def call(self, inputs, training=None, **kwargs):
        shortcut = self._shortcut_conv(inputs)
        shortcut = self._shortcut_bn(shortcut, training=training)

        x = self._conv1(inputs)
        x = self._bn1(x, training=training)
        x = tf.keras.activations.relu(x)

        x = self._conv2(x)
        x = self._bn2(x, training=training)
        x = tf.keras.activations.relu(x)

        x = self._conv3(x)
        x = self._bn3(x, training=training)
        x = tf.keras.activations.relu(x)

        x = x + shortcut
        x = self._upsample(x)

        if self._add_relu:
            x = tf.keras.activations.relu(x)

        return x

class BlockConv(tf.keras.layers.Layer):
    def __init__(self, num_filters):
        """
        Single block of common Conv4 architecture from https://arxiv.org/abs/1703.03400.
        :param num_filters:
        """
        super(BlockConv, self).__init__()

        self._num_filters = num_filters

    def build(self, input_shape):
        self._conv1 = tf.keras.layers.Conv2D(self._num_filters, kernel_size=3, strides=1, padding="SAME")
        self._bnorm1 = tf.keras.layers.BatchNormalization()
        self._pool1 = tf.keras.layers.MaxPool2D()  # No actual trainable weights

    def call(self, inputs, training=None, **kwargs):
        x = self._conv1(inputs)
        x = self._bnorm1(x, training=training)
        x = tf.keras.activations.relu(x)
        x = self._pool1(x)

        return x

class BlockConvReverse(tf.keras.layers.Layer):
    def __init__(self, num_filters):
        """
        Reverse, decoding block for the BlockConv layer.
        :param num_filters:
        """
        super(BlockConvReverse, self).__init__()

        self._num_filters = num_filters

    def build(self, input_shape):
        self._conv = tf.keras.layers.Conv2DTranspose(self._num_filters, kernel_size=3, strides=1, padding="SAME")
        self._bnorm = tf.keras.layers.BatchNormalization()
        self._upsample = tf.keras.layers.UpSampling2D(size=2)

    def call(self, inputs, training=None, **kwargs):
        x = self._conv(inputs)
        x = self._bnorm(x, training=training)
        x = tf.keras.activations.relu(x)
        x = self._upsample(x)

        return x
