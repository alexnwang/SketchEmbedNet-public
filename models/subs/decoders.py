import tensorflow as tf
from absl import logging

from models.subs.cells import LayerNormLSTMCell, HyperLSTMCell
from util import get_mixture_coef


class DecoderDefault(tf.keras.Model):
    def __init__(self, rnn_output_size, num_mixtures, rnn_model, cell_configs):
        """
        Autoregressive RNN Sequence decoder that outputs a sequence of points and pen states that constitute a sketch.

        Trained with teacher forcing during train time, the input at each timestep the ground truth data.
        During test time, we sample from the output distribution to obtain the input for the next timestep.
        :param rnn_output_size:
        :param num_mixtures:
        :param rnn_model:
        :param cell_configs:
        """
        super(DecoderDefault, self).__init__()

        self._rnn_model = rnn_model
        self._rnn_output_size = rnn_output_size
        self._num_mixtures = num_mixtures

        self._cell_configs = cell_configs

        if self._rnn_model == "hyper":
            self.cell = HyperLSTMCell(self._rnn_output_size,
                                      hyper_num_units=self._cell_configs["hyper_num_units"],
                                      hyper_embedding_size=self._cell_configs["hyper_embedding_size"],
                                      use_recurrent_dropout=self._cell_configs["use_recurrent_dropout"],
                                      recurrent_dropout_prob=self._cell_configs["recurrent_dropout_prob"])
        else:
            logging.fatal("Invalid RNN Cell Selection: %s", self._rnn_model)

    def build(self, input_shape):
        self._RNN = tf.keras.layers.RNN(self.cell)

        # 6 parameters per mixutre (pi, mu1, mu2, sigma1, sigma2, rho) + 3 pen state outputs
        glorot_init = tf.initializers.GlorotNormal()
        self._rnn_output_linear = tf.keras.layers.Dense(3 + self._num_mixtures * 6,
                                                        kernel_initializer=glorot_init, bias_initializer=glorot_init)

    def call(self, inputs, training=None, mask=None):
        ground_truth_sketch, z, init_cell_state, sequence_length = inputs

        batch_size = ground_truth_sketch.shape[0]

        time_dim_z = tf.reshape(z, [z.shape[0], 1, z.shape[1]])
        tiled_z = tf.tile(time_dim_z, [1, ground_truth_sketch.shape[1], 1])
        decoder_input = tf.concat([ground_truth_sketch, tiled_z], 2)

        loop_state_array = tf.TensorArray(dtype=tf.float32, size=sequence_length)

        def loop_fn(time, cell_output, cell_state, loop_state):
            emit_output = cell_output
            finished = (time >= sequence_length)

            if cell_output is None:
                next_cell_state = init_cell_state
                next_input = decoder_input[:, 0, :]

                next_loop_state = loop_state if training else loop_state_array
            else:
                next_cell_state = cell_state

                if training:
                    next_input = tf.cond(finished,
                                         lambda: tf.zeros(decoder_input[:, 0, :].shape),
                                         lambda: decoder_input[:, time, :])
                    next_loop_state = loop_state
                else:
                    step_pi, step_mu1, step_mu2, step_sigma1, step_sigma2, step_rho, step_pen, _ = get_mixture_coef(
                        self._rnn_output_linear(cell_output))

                    # Highest weighted mixture component
                    max_preidx = tf.math.argmax(step_pi, axis=1, output_type=tf.int32)
                    max_idx = tf.stack([tf.range(batch_size), max_preidx], axis=-1)
                    pi, mu1, mu2, sigma1, sigma2, rho = [tf.gather_nd(param, max_idx)
                                                         for param in [step_pi, step_mu1, step_mu2, step_sigma1, step_sigma2, step_rho]]

                    # Sample from bivariate gaussian
                    loc = tf.stack((mu1, mu2), axis=1)
                    cov = tf.stack((tf.stack((sigma1, tf.zeros(sigma1.shape)), axis=-1),
                                    tf.stack((rho * sigma2, sigma2 * tf.sqrt(1 - rho ** 2 + 1e-6)), axis=-1)),
                                   axis=-2)
                    eps = tf.random.normal(loc.shape)
                    xy = loc + tf.einsum("ijk,ik->ij", cov, eps)

                    # Convert softmax to one-hot
                    pen_one_hot = tf.one_hot(tf.argmax(step_pen, axis=1), depth=3)
                    stroke = tf.cast(tf.concat((xy, pen_one_hot), axis=1), dtype=tf.float32)

                    next_input = tf.concat([stroke, z], 1)
                    next_loop_state = loop_state.write(time - 1, stroke)

            return finished, next_input, next_cell_state, emit_output, next_loop_state

        emit_outputs_arr, final_state, loop_state_output_arr = tf.compat.v1.nn.raw_rnn(self.cell, loop_fn)

        param_output = tf.transpose(emit_outputs_arr.stack(), (1, 0, 2))
        raw_params = self._rnn_output_linear(tf.reshape(param_output, (batch_size * sequence_length, param_output.shape[-1])))
        params = [tf.cast(param, dtype=tf.float32) for param in get_mixture_coef(raw_params)]

        # If in training mode, only return parameters for NLL loss.
        # If in training mode, strokes are sampled at each timestep and are returned.
        if training:
            return params
        else:
            stroke_output = tf.transpose(loop_state_output_arr.stack(), (1, 0, 2))

            start_strokes = ground_truth_sketch[:, 0:1, :]
            return params, tf.concat((start_strokes, stroke_output), axis=1)

    def call_with_hyper_states(self, inputs, training=None, mask=None):
        """
        Alternative implementation of the "call" function.
        Saves and returns the hyperembedding activations per time step, used for the hyper_embedding_experiment.
        :param inputs:
        :param training:
        :param mask:
        :return:
        """
        ground_truth_sketch, z, init_cell_state, sequence_length = inputs

        batch_size = ground_truth_sketch.shape[0]

        time_dim_z = tf.reshape(z, [z.shape[0], 1, z.shape[1]])
        tiled_z = tf.tile(time_dim_z, [1, ground_truth_sketch.shape[1], 1])
        decoder_input = tf.concat([ground_truth_sketch, tiled_z], 2)

        loop_state_array = tf.TensorArray(dtype=tf.float32, size=sequence_length)

        def loop_fn(time, cell_output, cell_state, loop_state):
            emit_output = cell_output
            finished = (time >= sequence_length)

            if cell_output is None:
                next_cell_state = init_cell_state
                next_input = decoder_input[:, 0, :]

                next_loop_state = loop_state if training else loop_state_array
            else:
                next_cell_state = cell_state

                if training:
                    next_input = tf.cond(finished,
                                         lambda: tf.zeros(decoder_input[:, 0, :].shape),
                                         lambda: decoder_input[:, time, :])
                    next_loop_state = loop_state
                else:
                    step_pi, step_mu1, step_mu2, step_sigma1, step_sigma2, step_rho, step_pen, _ = get_mixture_coef(
                        self._rnn_output_linear(cell_output))

                    max_preidx = tf.math.argmax(step_pi, axis=1, output_type=tf.int32)
                    max_idx = tf.stack([tf.range(batch_size), max_preidx], axis=-1)

                    pi, mu1, mu2, sigma1, sigma2, rho = [tf.gather_nd(param, max_idx)
                                                         for param in [step_pi, step_mu1, step_mu2, step_sigma1, step_sigma2, step_rho]]

                    loc = tf.stack((mu1, mu2), axis=1)
                    cov = tf.stack((tf.stack((sigma1, tf.zeros(sigma1.shape)), axis=-1),
                                    tf.stack((rho * sigma2, sigma2 * tf.sqrt(1 - rho ** 2 + 1e-6)), axis=-1)),
                                   axis=-2)
                    eps = tf.random.normal(loc.shape)
                    xy = loc + tf.einsum("ijk,ik->ij", cov, eps)

                    pen_one_hot = tf.one_hot(tf.argmax(step_pen, axis=1), depth=3)
                    stroke = tf.cast(tf.concat((xy, pen_one_hot), axis=1), dtype=tf.float32)

                    next_input = tf.concat([stroke, z], 1)
                    next_loop_state = loop_state.write(time - 1, tf.concat((stroke, cell_state), axis=1))

            return finished, next_input, next_cell_state, emit_output, next_loop_state

        emit_outputs_arr, final_state, loop_state_output_arr = tf.compat.v1.nn.raw_rnn(self.cell, loop_fn)

        param_output = tf.transpose(emit_outputs_arr.stack(), (1, 0, 2))
        raw_params = self._rnn_output_linear(tf.reshape(param_output, (batch_size * sequence_length, param_output.shape[-1])))
        params = [tf.cast(param, dtype=tf.float32) for param in get_mixture_coef(raw_params)]

        if training:
            return params
        else:
            state_outputs = tf.transpose(loop_state_output_arr.stack(), (1, 0, 2))

            stroke_output = state_outputs[:, :, :5]
            cell_states = state_outputs[:, :, 5:]

            start_strokes = ground_truth_sketch[:, 0:1, :]
            return params, tf.concat((start_strokes, stroke_output), axis=1), cell_states
