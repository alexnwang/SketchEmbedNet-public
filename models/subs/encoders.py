import tensorflow as tf

from models.subs.conv_block import ResNet12Block


class EncoderConv(tf.keras.Model):
    def __init__(self, z_size, cell_state_size, filters=(4, 4, 8, 8, 8, 8)):
        """
        Minimal pure-conv encoder used in https://arxiv.org/pdf/1709.04121.pdf
        :param z_size:
        :param cell_state_size:
        :param filters:
        """
        super(EncoderConv, self).__init__()
        self._z_size = z_size

        glorot_init = tf.initializers.GlorotNormal()

        self._conv1 = tf.keras.layers.Conv2D(filters[0], 2, (2, 2), padding="SAME", kernel_initializer=glorot_init)
        self._conv2 = tf.keras.layers.Conv2D(filters[1], 2, padding="SAME", kernel_initializer=glorot_init)
        self._conv3 = tf.keras.layers.Conv2D(filters[2], 2, (2, 2), padding="SAME", kernel_initializer=glorot_init)
        self._conv4 = tf.keras.layers.Conv2D(filters[3], 2, padding="SAME", kernel_initializer=glorot_init)
        self._conv5 = tf.keras.layers.Conv2D(filters[2], 2, (2, 2), padding="SAME", kernel_initializer=glorot_init)
        self._conv6 = tf.keras.layers.Conv2D(filters[3], 2, padding="SAME", kernel_initializer=glorot_init)

        random_init = tf.initializers.RandomNormal(0.001)
        zero_init = tf.initializers.Zeros()

        self._mu = tf.keras.layers.Dense(z_size, kernel_initializer=random_init, bias_initializer=zero_init)
        self._var = tf.keras.layers.Dense(z_size, kernel_initializer=random_init, bias_initializer=zero_init)

        self._cell_init_state = tf.keras.layers.Dense(cell_state_size, activation=tf.keras.activations.tanh,
                                                      bias_initializer=zero_init, kernel_initializer=random_init)

    def call(self, inputs, training=None, **kwargs):
        x = self._conv1(inputs)
        x = tf.keras.activations.relu(x)

        x = self._conv2(x)
        x = tf.keras.activations.relu(x)

        x = self._conv3(x)
        x = tf.keras.activations.relu(x)

        x = self._conv4(x)
        x = tf.keras.activations.relu(x)

        x = self._conv5(x)
        x = tf.keras.activations.relu(x)

        x = self._conv6(x)
        x = tf.keras.activations.tanh(x)

        x = tf.reshape(x, (inputs.shape[0], -1))

        mu = self._mu(x)
        logvar = self._var(x)

        sigma = tf.exp(logvar / 2.0)

        z = mu + tf.multiply(tf.random.normal(mu.shape), sigma)

        cell_init_state = self._cell_init_state(z)

        return z, cell_init_state, mu, logvar

class EncoderBlock4(tf.keras.Model):
    def __init__(self, z_size, cell_state_size, filters=(64, 64, 64, 64)):
        """
        Conv4 encoder from https://arxiv.org/abs/1703.03400
        :param z_size:
        :param cell_state_size:
        :param filters:
        """
        super(EncoderBlock4, self).__init__()

        self._cell_state_size = cell_state_size
        self._filters = filters
        self._z_size = z_size

    def build(self, input_shape):
        glorot_init = tf.initializers.GlorotNormal()
        random_init = tf.initializers.RandomNormal(0.001)
        zero_init = tf.initializers.Zeros()

        self._conv1 = tf.keras.layers.Conv2D(self._filters[0], 3, padding="SAME", kernel_initializer=glorot_init)
        self._bnorm1 = tf.keras.layers.BatchNormalization()
        self._pool1 = tf.keras.layers.MaxPool2D()  # No actual trainable weights

        self._conv2 = tf.keras.layers.Conv2D(self._filters[1], 3, padding="SAME", kernel_initializer=glorot_init)
        self._bnorm2 = tf.keras.layers.BatchNormalization()
        self._pool2 = tf.keras.layers.MaxPool2D()

        self._conv3 = tf.keras.layers.Conv2D(self._filters[2], 3, padding="SAME", kernel_initializer=glorot_init)
        self._bnorm3 = tf.keras.layers.BatchNormalization()
        self._pool3 = tf.keras.layers.MaxPool2D()

        self._conv4 = tf.keras.layers.Conv2D(self._filters[3], 3, padding="SAME", kernel_initializer=glorot_init)
        self._bnorm4 = tf.keras.layers.BatchNormalization()
        self._pool4 = tf.keras.layers.MaxPool2D()

        self._final = tf.keras.layers.GlobalAveragePooling2D()

        self._mu = tf.keras.layers.Dense(self._z_size, kernel_initializer=random_init, bias_initializer=zero_init)
        self._var = tf.keras.layers.Dense(self._z_size, kernel_initializer=random_init, bias_initializer=zero_init)

        self._cell_init_state = tf.keras.layers.Dense(self._cell_state_size, activation=tf.keras.activations.tanh,
                                                      bias_initializer=zero_init, kernel_initializer=random_init)

    def call(self, inputs, training=None, **kwargs):
        x = self._conv1(inputs)
        x = self._bnorm1(x, training=training)
        x = tf.keras.activations.relu(x)
        x = self._pool1(x)

        x = self._conv2(x)
        x = self._bnorm2(x, training=training)
        x = tf.keras.activations.relu(x)
        x = self._pool2(x)

        x = self._conv3(x)
        x = self._bnorm3(x, training=training)
        x = tf.keras.activations.relu(x)
        x = self._pool3(x)

        x = self._conv4(x)
        x = self._bnorm4(x, training=training)
        x = tf.keras.activations.relu(x)
        x = self._pool4(x)

        x = self._final(x)

        mu = self._mu(x)
        logvar = self._var(x)
        sigma = tf.exp(logvar / 2.0)

        z = mu + tf.multiply(tf.random.normal(mu.shape), sigma)

        cell_init_state = self._cell_init_state(z)

        return z, cell_init_state, mu, logvar

class EncoderResnet12(tf.keras.Model):
    def __init__(self, z_size, cell_state_size, filters=(64, 128, 256, 512)):
        """
        Encoder using ResNet12.
        :param z_size:
        :param cell_state_size:
        :param filters:
        """
        super(EncoderResnet12, self).__init__()

        self._cell_state_size = cell_state_size
        self._filters = filters
        self._z_size = z_size

    def build(self, input_shape):
        random_init = tf.initializers.RandomNormal(0.001)
        zero_init = tf.initializers.Zeros()

        self._resnet_block1 = ResNet12Block(self._filters[0])
        self._resnet_block2 = ResNet12Block(self._filters[1])
        self._resnet_block3 = ResNet12Block(self._filters[2])
        self._resnet_block4 = ResNet12Block(self._filters[3])

        self._final = tf.keras.layers.GlobalAveragePooling2D()

        self._mu = tf.keras.layers.Dense(self._z_size, kernel_initializer=random_init, bias_initializer=zero_init)
        self._var = tf.keras.layers.Dense(self._z_size, kernel_initializer=random_init, bias_initializer=zero_init)

        self._cell_init_state = tf.keras.layers.Dense(self._cell_state_size, activation=tf.keras.activations.tanh,
                                                      bias_initializer=zero_init, kernel_initializer=random_init)

    def call(self, inputs, training=None, **kwargs):
        x = self._resnet_block1(inputs, training=training)
        x = self._resnet_block2(x, training=training)
        x = self._resnet_block3(x, training=training)
        x = self._resnet_block4(x, training=training)

        x = self._final(x)

        mu = self._mu(x)
        logvar = self._var(x)
        sigma = tf.exp(logvar / 2.0)

        z = mu + tf.multiply(tf.random.normal(mu.shape), sigma)

        cell_init_state = self._cell_init_state(z)

        return z, cell_init_state, mu, logvar
