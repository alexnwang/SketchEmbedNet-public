import tensorflow as tf


class LayerNormLSTMCell(tf.keras.layers.AbstractRNNCell):
    """Layer-Norm, with Ortho Init. and Recurrent Dropout without Memory Loss.

    https://arxiv.org/abs/1607.06450 - Layer Norm
    https://arxiv.org/abs/1603.05118 - Recurrent Dropout without Memory Loss
    """

    def __init__(self,
                 num_units,
                 forget_bias=1.0,
                 use_recurrent_dropout=False,
                 dropout_keep_prob=0.90,
                 **kwargs):
        """Initialize the Layer Norm LSTM cell.

        Args:
          num_units: int, The number of units in the LSTM cell.
          forget_bias: float, The bias added to forget gates (default 1.0).
          use_recurrent_dropout: Whether to use Recurrent Dropout (default False)
          dropout_keep_prob: float, dropout keep probability (default 0.90)
        """
        super(LayerNormLSTMCell, self).__init__(**kwargs)

        self._num_units = num_units
        self._forget_bias = forget_bias
        self._use_recurrent_dropout = use_recurrent_dropout
        self._dropout_keep_prob = dropout_keep_prob

        ortho_initializer = tf.initializers.Orthogonal()
        glorot_initializer = tf.initializers.GlorotNormal()

        # Original implementation split this layer into w_xh and w_hh, however we anticipate that using ortho only may be better.
        self._input_w_xh = tf.keras.layers.Dense(units=4 * self._num_units, use_bias=False, kernel_initializer=glorot_initializer)
        self._hidden_w_hh = tf.keras.layers.Dense(units=4 * self._num_units, use_bias=False, kernel_initializer=ortho_initializer)
        self._layer_norm = tf.keras.layers.LayerNormalization(axis=1)

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units * 2

    def call(self, x, state, timestep=0, scope=None):
        hidden, cell_state = tf.split(state, 2, 1)
        gates = tf.split(self._input_w_xh(x) + self._hidden_w_hh(hidden), 4, 1)
        i, j, f, o = [self._layer_norm(gate) for gate in gates]

        if self._use_recurrent_dropout:
            g = tf.keras.layers.Dropout(tf.tanh(j), self._dropout_keep_prob)
        else:
            g = tf.tanh(j)

        new_cell_state = cell_state * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * g
        new_hidden_state = tf.tanh(self._layer_norm(new_cell_state)) + tf.sigmoid(o)

        next_state = tf.concat((new_hidden_state, new_cell_state), 1)
        return new_cell_state, next_state


class HyperLSTMCell(tf.keras.layers.AbstractRNNCell):
    """HyperLSTM with Ortho Init, Layer Norm, Recurrent Dropout, no Memory Loss.

    https://arxiv.org/abs/1609.09106
    http://blog.otoro.net/2016/09/28/hyper-networks/
    """

    def __init__(self,
                 num_units,
                 forget_bias=1.0,
                 use_recurrent_dropout=False,
                 recurrent_dropout_prob=0.90,
                 use_layer_norm=True,
                 hyper_forget_bias=1.0,
                 hyper_num_units=256,
                 hyper_embedding_size=32,
                 hyper_use_recurrent_dropout=False,
                 **kwargs):
        """Initialize the Layer Norm HyperLSTM cell.

        Args:
          num_units: int, The number of units in the LSTM cell.
          forget_bias: float, The bias added to forget gates (default 1.0).
          use_recurrent_dropout: Whether to use Recurrent Dropout (default False)
          recurrent_dropout_prob: float, dropout keep probability (default 0.90)
          use_layer_norm: boolean. (default True)
            Controls whether we use LayerNorm layers in main LSTM & HyperLSTM cell.
          hyper_num_units: int, number of units in HyperLSTM cell.
            (default is 128, recommend experimenting with 256 for larger tasks)
          hyper_embedding_size: int, size of signals emitted from HyperLSTM cell.
            (default is 16, recommend trying larger values for large datasets)
          hyper_use_recurrent_dropout: boolean. (default False)
            Controls whether HyperLSTM cell also uses recurrent dropout.
            Recommend turning this on only if hyper_num_units becomes large (>= 512)
        """
        super(HyperLSTMCell, self).__init__(**kwargs)

        # ----- Parameters ----- #
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._use_recurrent_dropout = use_recurrent_dropout
        self._dropout_keep_prob = recurrent_dropout_prob
        self._use_layer_norm = use_layer_norm
        self._hyper_forget_bias = hyper_forget_bias
        self._hyper_num_units = hyper_num_units
        self._hyper_embedding_size = hyper_embedding_size
        self._hyper_use_recurrent_dropout = hyper_use_recurrent_dropout

        self._total_num_units = self._num_units + self._hyper_num_units

        # ----- Build Model Components ----- #
        self._hyper_cell = LayerNormLSTMCell(num_units=self._hyper_num_units,
                                             forget_bias=self._hyper_forget_bias,
                                             use_recurrent_dropout=self._hyper_use_recurrent_dropout,
                                             dropout_keep_prob=self._dropout_keep_prob)

        ortho_initializer = tf.initializers.Orthogonal()
        glorot_initializer = tf.initializers.GlorotNormal()
        #
        # Original implementation split this layer into w_xh and w_hh, however we anticipate that using ortho only may be better.
        self._input_wxh = tf.keras.layers.Dense(units=4 * self._num_units, use_bias=False, kernel_initializer=glorot_initializer)
        self._hidden_whh = tf.keras.layers.Dense(4 * self._num_units, use_bias=False, kernel_initializer=ortho_initializer)

        self._bias = tf.Variable(tf.zeros(4 * self._num_units))

        # recurrent batch norm init trick (https://arxiv.org/abs/1603.09025).
        self._hnorm_input_ix = HyperNorm(num_units=self._num_units, embedding_size=self._hyper_embedding_size, use_bias=False)
        self._hnorm_input_jx = HyperNorm(num_units=self._num_units, embedding_size=self._hyper_embedding_size, use_bias=False)
        self._hnorm_input_fx = HyperNorm(num_units=self._num_units, embedding_size=self._hyper_embedding_size, use_bias=False)
        self._hnorm_input_ox = HyperNorm(num_units=self._num_units, embedding_size=self._hyper_embedding_size, use_bias=False)

        self._hnorm_hidden_ih = HyperNorm(num_units=self._num_units, embedding_size=self._hyper_embedding_size, use_bias=True)
        self._hnorm_hidden_jh = HyperNorm(num_units=self._num_units, embedding_size=self._hyper_embedding_size, use_bias=True)
        self._hnorm_hidden_fh = HyperNorm(num_units=self._num_units, embedding_size=self._hyper_embedding_size, use_bias=True)
        self._hnorm_hidden_oh = HyperNorm(num_units=self._num_units, embedding_size=self._hyper_embedding_size, use_bias=True)

        self._layer_norm = tf.keras.layers.LayerNormalization()
        self._layer_norm2 = tf.keras.layers.LayerNormalization()

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return 2 * self._total_num_units

    def call(self, input, state):
        joint_h, joint_c = tf.split(state, 2, 1)

        hidden = joint_h[:, 0:self._num_units]
        cell = joint_c[:, 0:self._num_units]

        hyper_state = tf.concat((joint_h[:, self._num_units:], joint_c[:, self._num_units:]),
                                axis=1)

        hyper_input = tf.concat((input, hidden), 1)
        hyper_output, hyper_new_state = self._hyper_cell(hyper_input, hyper_state)

        x_w = self._input_wxh(input)
        hidden_w = self._hidden_whh(hidden)

        ix, jx, fx, ox = tf.split(hidden_w, 4, 1)
        ix = self._hnorm_input_ix(hyper_output, ix)
        jx = self._hnorm_input_jx(hyper_output, jx)
        fx = self._hnorm_input_fx(hyper_output, fx)
        ox = self._hnorm_input_ox(hyper_output, ox)

        ih, jh, fh, oh = tf.split(x_w, 4, 1)
        ih = self._hnorm_hidden_ih(hyper_output, ih)
        jh = self._hnorm_hidden_jh(hyper_output, jh)
        fh = self._hnorm_hidden_fh(hyper_output, fh)
        oh = self._hnorm_hidden_oh(hyper_output, oh)

        ib, jb, fb, ob = tf.split(self._bias, 4, 0)
        i = ix + ih + ib
        j = jx + jh + jb
        f = fx + fh + fb
        o = ox + oh + ob

        concat = tf.concat((i, j, f, o), 1)
        i, j, f, o = tf.split(self._layer_norm(concat), 4, 1)

        if self._use_recurrent_dropout:
            g = tf.nn.dropout(tf.tanh(j), self._dropout_keep_prob)
        else:
            g = tf.tanh(j)

        new_cell_state = cell * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * g
        new_hidden_state = tf.tanh(self._layer_norm2(new_cell_state)) * tf.sigmoid(o)

        new_hyper_hidden, new_hyper_cell = tf.split(hyper_new_state, 2, 1)
        new_hidden = tf.concat((new_hidden_state, new_hyper_hidden), 1)
        new_cell = tf.concat((new_cell_state, new_hyper_cell), 1)
        new_total_state = tf.concat((new_hidden, new_cell), 1)

        return new_hidden_state, new_total_state


class HyperNorm(tf.keras.layers.Layer):
    def __init__(self, num_units, embedding_size, use_bias, **kwargs):
        super(HyperNorm, self).__init__(**kwargs)

        self._num_units = num_units
        self._embedding_size = embedding_size
        self._use_bias = use_bias


        # ----- Build Model ----- #
        # recurrent batch norm init trick (https://arxiv.org/abs/1603.09025).
        init_gamma = 0.10

        zero_init = tf.initializers.Zeros()
        const_init = tf.initializers.Constant(value=1.0)

        gamma_init = tf.initializers.Constant(value=init_gamma/self._embedding_size)

        self._zw = tf.keras.layers.Dense(self._embedding_size, activation=None, use_bias=True,
                                         kernel_initializer=zero_init, bias_initializer=const_init)
        self._alpha = tf.keras.layers.Dense(self._num_units, activation=None, use_bias=False,
                                            kernel_initializer=gamma_init)

        if self._use_bias:
            gaussian_init = tf.initializers.RandomNormal(stddev=0.01)
            self._zb = tf.keras.layers.Dense(self._embedding_size, activation=None, use_bias=False,
                                             kernel_initializer=gaussian_init)
            self._beta = tf.keras.layers.Dense(self._num_units, activation=None,
                                               kernel_initializer=const_init)

    def call(self, hyper_output, layer):
        zw = self._zw(hyper_output)
        alpha = self._alpha(zw)
        result = alpha * layer

        if self._use_bias:
            result += self._beta(self._zb(hyper_output))

        return result
