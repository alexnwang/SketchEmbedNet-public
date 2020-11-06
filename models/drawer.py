import os

import tensorflow as tf
import numpy as np
from PIL import Image
from absl import logging
from time import time

from models.base import TrainableModel, register_model
from models.subs.decoders import DecoderDefault
from models.subs.encoders import EncoderConv
from util import compute_pen_state_loss, compute_mdn_loss, process_write_out, compute_pixel_loss, strokes_to_image, \
    bilinear_interpolate_4_vectors, stroke_three_format, scale_and_rasterize
from util.write_routines import parallel_writer_sketches

try:
    import horovod.tensorflow as hvd
except:
    hvd = None

@register_model('drawer')
class DrawerModel(TrainableModel):
    def __init__(self, base_dir, model_id, params, training=True, ckpt=None):
        """
        SketchEmbedding drawing model.
        :param params:
        :param name:
        """
        # ----- Model Parameters ----- #
        self._z_size = params.z_size
        self._num_mixture = params.num_mixture
        self._dec_rnn_size = params.rnn_output_size
        self._rnn_model = params.rnn_cell

        self._cell_configs = params.cell_configs

        self._kl_tolerance = params.kl_tolerance
        self._kl_weight = params.kl_weight

        self._pixel_loss_weight_max = params.pixel_loss_weight_max
        self._pixel_loss_weight_min = params.pixel_loss_weight_min
        self._pixel_loss_weight_interval = params.pixel_loss_weight_interval
        self._pixel_loss_step = params.pixel_loss_step
        self._pixel_loss_weight = tf.Variable(self._pixel_loss_weight_min, trainable=False, dtype=tf.float32)

        self._sigma_decay_init = params.sigma_init
        self._sigma_decay_start = params.sigma_decay_start
        self._sigma_decay_freq = params.sigma_decay_freq
        self._sigma_decay_rate = params.sigma_decay_rate
        self._sigma = tf.Variable(self._sigma_decay_init, trainable=False, dtype=tf.float32)

        # ----- Training Parameters ----- #
        self._lr = params.lr
        self._lr_decay_freq = params.lr_decay_freq
        self._lr_decay_rate = params.lr_decay_rate
        self._gradient_cap = params.gradient_cap

        # ----- Other Parameters ----- #
        self._distributed = params.distributed

        if self._distributed:
            # Modify scheduling if distributing
            self._pixel_loss_step = self._pixel_loss_step // hvd.size()
            self._sigma_decay_freq = self._sigma_decay_freq // hvd.size()
            self._lr_decay_freq = self._lr_decay_freq // hvd.size()

        # ----- Init Model ----- #
        super(DrawerModel, self).__init__(base_dir, model_id, training, ckpt=ckpt)

    def _build_model(self):
        self._decoder = DecoderDefault(self._dec_rnn_size, self._num_mixture, self._rnn_model, self._cell_configs)
        self._encoder = EncoderConv(self._z_size, self._decoder.cell.state_size)

        lr_init = self._lr if not self._distributed else self._lr * hvd.size()
        lr = tf.keras.optimizers.schedules.ExponentialDecay(lr_init, self._lr_decay_freq, self._lr_decay_rate)
        self._optimizer = tf.optimizers.Adam(learning_rate=lr, clipvalue=self._gradient_cap)

    def _checkpoint_model(self):
        ckpt = tf.train.Checkpoint(optimizer=self._optimizer,
                                   encoder=self._encoder,
                                   decoder=self._decoder)
        self._ckpt_manager = tf.train.CheckpointManager(ckpt, self._checkpoint_dir, max_to_keep=None)

        if self._ckpt:
            filtered_ckpts = list(filter(lambda x: x.endswith("-" + str(self._ckpt)), self._ckpt_manager.checkpoints))
            if len(filtered_ckpts) == 1:
                logging.info("Checkpoint %s found. Restoring.", str(filtered_ckpts[0]))
                status = ckpt.restore(filtered_ckpts[0])
                if self.training:
                    status.assert_existing_objects_matched()
                else:
                    status.expect_partial()
            else:
                logging.fatal("%s matching checkpoints found. Exiting.", len(filtered_ckpts))
        elif self._ckpt_manager.latest_checkpoint:
            logging.info("Restoring Checkpoint: %s", self._ckpt_manager.latest_checkpoint)
            status = ckpt.restore(self._ckpt_manager.latest_checkpoint)
            if self.training:
                status.assert_existing_objects_matched()
            else:
                status.expect_partial()

    def train(self, train_dataset, train_steps, print_freq, save_freq, eval_dataset=None, eval_freq=None):
        if self._distributed:
            train_steps = train_steps // hvd.size()
            print_freq = print_freq // hvd.size()
            save_freq = save_freq // hvd.size()
            if eval_freq:
                eval_freq = eval_freq // hvd.size()

        if eval_dataset and not eval_freq:
            eval_freq = save_freq

        train_dataset, _ = train_dataset  # The second element is a saveable file-based dataset, currently not used
        train_iter = train_dataset.__iter__()

        last_time = time()
        start_time = last_time
        first_step = True
        for step in tf.range(self._optimizer.iterations + 1, tf.constant(train_steps + 1)):
            y_sketch_gt, y_sketch_teacher, x_image = next(train_iter)[0:3]

            total_loss, pen_loss, offset_loss, pixel_loss, kl_loss = self.train_step(y_sketch_gt, y_sketch_teacher, x_image, first_step)
            try:
                tf.debugging.check_numerics(total_loss, "NaN Loss found. Reverting to previous checkpoint.")
            except:
                self._ckpt = None
                self._checkpoint_model()  # Reload most recent checkpoint
                raise

            if (not self._distributed) or (hvd.rank() == 0):
                if step and step % print_freq == 0:
                    curr_time = time()
                    logging.info("Step: %6d | Loss: %.5f | PenL: %.5f | OffsetL(%4.2f): %.5f | PixelL(%4.2f): %.4f | KLL: %.4f | LR: %.5f | time/step: %.4f | Total Time: %7d",
                                 step * (1 if not self._distributed else hvd.size()), total_loss, pen_loss,
                                 1 - self._pixel_loss_weight.numpy(), offset_loss,
                                 self._pixel_loss_weight.numpy(), pixel_loss,
                                 kl_loss, self._optimizer._decayed_lr('float32').numpy(),
                                 (curr_time-last_time)/(print_freq * (1 if not self._distributed else hvd.size())), curr_time-start_time)
                    last_time = curr_time

                    with self._writer.as_default():
                        self._write_summaries(step, {"lr": self._optimizer._decayed_lr('float32'),
                                                     "train_loss": total_loss, "pen_loss": pen_loss,
                                                     "pixel_loss": pixel_loss, "offset_loss": offset_loss,
                                                     "pixel_loss_weight": self._pixel_loss_weight,
                                                     "weighted_pixel_loss": self._pixel_loss_weight.numpy() * pixel_loss,
                                                     "weighted_offset_loss": (1-self._pixel_loss_weight.numpy()) * offset_loss,
                                                     "sigma": self._sigma.numpy()})
                if step and step % save_freq == 0:
                    self._ckpt_manager.save(step * (1 if not self._distributed else hvd.size()))

                if eval_dataset and step and step % eval_freq == 0:
                    self.evaluate(step * (1 if not self._distributed else hvd.size()), eval_dataset)

    def evaluate(self, step, eval_dataset):
        eval_dataset, _ = eval_dataset
        total_loss_mean, pen_loss_mean, offset_loss_mean, pixel_loss_mean, kl_loss_mean = (tf.keras.metrics.Mean(), tf.keras.metrics.Mean(),
                                                                                           tf.keras.metrics.Mean(), tf.keras.metrics.Mean(),
                                                                                           tf.keras.metrics.Mean())
        eval_start_time = time()
        for entry in eval_dataset.__iter__():
            y_sketch_gt, y_sketch_teacher, x_image = entry[0:3]

            params = self.forward(y_sketch_teacher, x_image, training=False, generation_length=y_sketch_gt.shape[1]-1)[:-1]
            total_loss, pen_loss, offset_loss, pixel_loss, kl_loss = self.compute_loss(params, y_sketch_gt, x_image)

            total_loss_mean(total_loss)
            pen_loss_mean(pen_loss)
            offset_loss_mean(offset_loss)
            pixel_loss_mean(pixel_loss)
            kl_loss_mean(kl_loss)
        last_time = time()

        eval_total, eval_pen, eval_offset, eval_pixel, eval_kl = (total_loss_mean.result(), pen_loss_mean.result(),
                                                                  offset_loss_mean.result(), pixel_loss_mean.result(),
                                                                  kl_loss_mean.result())

        with self._writer.as_default():
            self._write_summaries(step, {"eval_loss": eval_total, "eval_pen": eval_pen, "eval_offset": eval_offset,
                                         "eval_pixel": eval_pixel, "eval_kl": eval_kl, "eval_pixel_pen": eval_pixel + eval_pen,
                                         "eval_weighted_pixel": self._pixel_loss_weight.numpy() * pixel_loss,
                                         "eval_weighted_offset": (1-self._pixel_loss_weight.numpy()) * offset_loss,
                                         "eval_sigma": self._sigma.numpy()})

        logging.info(
            "Eval Done   | Loss: %.5f | PenL: %.5f | OffsetL(%4.2f): %.5f | PixelL(%4.2f): %.4f | KLL: %.4f | Eval Time: %.4f",
            eval_total, eval_pen,
            1 - self._pixel_loss_weight.numpy(), eval_offset,
            self._pixel_loss_weight.numpy(), eval_pixel,
            eval_kl, last_time - eval_start_time)

    def test(self, test_dataset, result_name, steps=None, generation_length=64, decodes=1):
        logging.info("Beginning testing loop")
        sampling_dir = os.path.join(self._sampling_dir, result_name)
        test_dataset, _ = test_dataset

        # Begin Writing Child-Process
        process, write_queue = process_write_out(parallel_writer_sketches, (sampling_dir,))

        try:
            for step, entry in enumerate(test_dataset):
                if step == steps:
                    break

                if len(entry) == 2:
                    x_image, class_names = entry
                    y_sketch_gt, y_sketch_teacher = (tf.tile(tf.constant([[[0., 0., 1., 0., 0.]]]), (x_image.shape[0], 2, 1)),
                                                     tf.tile(tf.constant([[[0., 0., 1., 0., 0.]]]), (x_image.shape[0], 2, 1)))
                else:
                    y_sketch_gt, y_sketch_teacher, x_image, class_names = entry[0:4]

                if decodes == 1:
                    output = self.forward(y_sketch_teacher, x_image, training=False, generation_length=generation_length)[-1]
                    np_images, np_sketch, np_classes, np_prediction = (x_image.numpy(),
                                                                       y_sketch_gt.numpy(), class_names.numpy(), output.numpy())

                    for idx in range(x_image.shape[0]):
                        write_queue.put({"rasterized_images": np_images[idx],
                                         "stroke_five_sketches": np_sketch[idx],
                                         "class_names": np_classes[idx],
                                         "stroke_predictions": np_prediction[idx]})
                else:
                    z, _, _ = self.embed(x_image, training=False)
                    for _ in range(decodes):
                        _, output = self.decode(z, training=False, generation_length=generation_length)
                        np_images, np_sketch, np_classes, np_prediction = (x_image.numpy(),
                                                                           y_sketch_gt.numpy(), class_names.numpy(), output.numpy())

                        for idx in range(0, x_image.shape[0]):
                            write_queue.put({"rasterized_images": np_images[idx],
                                             "stroke_five_sketches": np_sketch[idx],
                                             "class_names": np_classes[idx],
                                             "stroke_predictions": np_prediction[idx]})

            write_queue.put(None)
        except:
            process.terminate()
            raise

        process.join()
        logging.info("Testing complete")

    @tf.function
    def train_step(self, y_sketch_gt, y_sketch_teacher, x_image, first_step):
        with tf.GradientTape() as tape:
            params = self.forward(y_sketch_teacher, x_image, training=True)[:-1]
            total_loss, pen_loss, offset_loss, pixel_loss, kl_loss = self.compute_loss(params, y_sketch_gt, x_image)

        if self._distributed:
            tape = hvd.DistributedGradientTape(tape)

        grads = tape.gradient(total_loss, self._encoder.trainable_variables + self._decoder.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self._encoder.trainable_variables + self._decoder.trainable_variables))

        if self._distributed and first_step:
            hvd.broadcast_variables(self._encoder.trainable_variables + self._decoder.trainable_variables, root_rank=0)
            hvd.broadcast_variables(self._optimizer.variables(), root_rank=0)

        return total_loss, pen_loss, offset_loss, pixel_loss, kl_loss

    @tf.function
    def compute_loss(self, params, stroke_gt, image_gt):
        step = tf.cast(self._optimizer.iterations + 1, tf.float32)
        self._pixel_loss_weight.assign(tf.minimum(self._pixel_loss_weight_max,
                                                  self._pixel_loss_weight_min +
                                                  tf.floor(tf.math.divide_no_nan(step, self._pixel_loss_step)) *
                                                  self._pixel_loss_weight_interval))
        self._sigma.assign(self._sigma_decay_init * tf.pow(self._sigma_decay_rate,
                                                           tf.floor(tf.maximum(0., step - self._sigma_decay_start) /
                                                                    self._sigma_decay_freq)))

        output_params, latent_mu, latent_logvar = params
        pi, mu1, mu2, sigma1, sigma2, rho, pen, pen_logits = output_params

        # Extract parameters
        stroke_gt_cut = stroke_gt[:, 1:, :]
        x1_gt, x2_gt, eos_data, eoc_data, cont_data = tf.split(tf.reshape(stroke_gt_cut, (-1, 5)), 5, 1)
        pen_gt = tf.concat([eos_data, eoc_data, cont_data], 1)
        batch_size = tf.shape(image_gt)[0]
        pixel_dims = tf.shape(image_gt)[1:3]

        # Compute losses
        pen_state_loss_batched = compute_pen_state_loss(pen_logits, pen_gt)
        mdn_loss_batched = compute_mdn_loss(pi, mu1, mu2, sigma1, sigma2, rho, x1_gt, x2_gt, pen_gt)
        pixel_loss_batched = compute_pixel_loss(pi, mu1, mu2, sigma1, sigma2, rho, pen, stroke_gt, batch_size, pixel_dims)

        # reduce loss vectors and compute total loss
        pen_state_loss = tf.reduce_mean(pen_state_loss_batched)
        mdn_loss = tf.reduce_mean(tf.boolean_mask(mdn_loss_batched, tf.math.is_finite(mdn_loss_batched)))
        pixel_loss = tf.reduce_mean(pixel_loss_batched)

        reconstruction_loss = (pen_state_loss +
                               (mdn_loss * (1 - self._pixel_loss_weight)) +
                               ((pixel_loss * self._pixel_loss_weight) if self._pixel_loss_weight > 0.1 else 0.))

        # Only compute KL Loss if it is being used.
        if self._kl_weight > 0.:
            kl_loss = -0.5 * tf.reduce_mean(1 + latent_logvar - tf.square(latent_mu) - tf.exp(latent_logvar))
            reconstruction_loss += (tf.maximum(kl_loss, self._kl_tolerance) - self._kl_tolerance) * self._kl_weight
        else:
            kl_loss = 0.

        return reconstruction_loss, pen_state_loss, mdn_loss, pixel_loss, kl_loss

    @tf.function
    def embed(self, x_image, training=False):
        """
        Embeds the given image
        :param x_image:
        :return: z, mu, logvar
        """
        z, _, mu, logvar = self._encoder(x_image, training=training)
        return z, mu, logvar

    @tf.function
    def forward(self, y_sketch, x_image, training, generation_length=0):
        input_sketch = y_sketch[:, :-1, :]
        if training:
            generation_length = input_sketch.shape[1]

        z, init_cell_state, mu, logvar = self._encoder(x_image, training)

        if training:
            params = self._decoder((input_sketch, z, init_cell_state, generation_length), training)
            strokes = None
        else:
            params, strokes = self._decoder((input_sketch, z, init_cell_state, generation_length), training)

        return params, mu, logvar, strokes

    @tf.function
    def decode(self, z, training, generation_length=0, with_hyper_states=False):
        init_cell_state = self._encoder._cell_init_state(z)
        input_sketch = tf.tile(tf.constant([[[0., 0., 1., 0., 0.]]]), (z.shape[0], 1, 1))

        if training:
            params = self._decoder((input_sketch, z, init_cell_state, generation_length), training)
            return params
        else:
            if not with_hyper_states:
                params, strokes = self._decoder((input_sketch, z, init_cell_state, generation_length), training)
                return params, strokes
            else:
                params, strokes, hyper_states = self._decoder.call_with_hyper_states((input_sketch, z, init_cell_state, generation_length),
                                                                                     training)
                return params, strokes, hyper_states
