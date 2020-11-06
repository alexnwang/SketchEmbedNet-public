import os
from time import time

import tensorflow as tf
from absl import logging

from models.base import TrainableModel, register_model
from models.subs.conv_block import ResNet12Block, ResNet12BlockReverse
from util import process_write_out
from util.write_routines import parallel_writer_vae_latent


@register_model("vae")
class VAE(TrainableModel):
    def __init__(self, base_dir, model_id, params, training=True):
        """
        Pixel-domain VAE model as a benchmark comparison for embedding learning purposes
        :param base_dir:
        :param model_id:
        :param params:
        :param training:
        """
        self._latent_size = params.latent_size
        self._png_dim = params.png_dim
        self._grayscale = params.grayscale

        self._kl_weight = params.kl_weight
        self._kl_tolerance = params.kl_tolerance

        self._filters = params.filters

        self._lr = params.lr
        self._lr_decay_step = params.lr_decay_step
        self._lr_decay_rate = params.lr_decay_rate

        super(VAE, self).__init__(base_dir, model_id, training)

    def _build_model(self):
        self._conv_encoder = tf.keras.Sequential([
            ResNet12Block(self._filters[0]),
            ResNet12Block(self._filters[1]),
            ResNet12Block(self._filters[2]),
            ResNet12Block(self._filters[3]),
            tf.keras.layers.GlobalAveragePooling2D()
        ])

        self._mu = tf.keras.layers.Dense(self._latent_size)
        self._var = tf.keras.layers.Dense(self._latent_size)

        self._conv_decoder = tf.keras.Sequential([
            tf.keras.layers.UpSampling2D((self._png_dim//16, self._png_dim//16)),
            ResNet12BlockReverse(self._filters[2]),
            ResNet12BlockReverse(self._filters[1]),
            ResNet12BlockReverse(self._filters[0]),
            ResNet12BlockReverse(1 if self._grayscale else 3)
        ])

        lr = tf.keras.optimizers.schedules.ExponentialDecay(self._lr, self._lr_decay_step, self._lr_decay_rate)
        self._optimizer = tf.optimizers.Adam(learning_rate=lr)

    def _checkpoint_model(self):
        ckpt = tf.train.Checkpoint(optimizer=self._optimizer,
                                   conv_encoder=self._conv_encoder,
                                   mean=self._mu,
                                   var=self._var,
                                   conv_decoder=self._conv_decoder)
        self._ckpt_manager = tf.train.CheckpointManager(ckpt, self._checkpoint_dir, max_to_keep=None)
        if self._ckpt_manager.latest_checkpoint:
            logging.info("Restoring Checkpoint: %s", self._ckpt_manager.latest_checkpoint)
            status = ckpt.restore(self._ckpt_manager.latest_checkpoint)
            if self.training:
                status.assert_existing_objects_matched()
            else:
                status.expect_partial()

    def train(self, train_dataset, train_steps, print_freq, save_freq, eval_dataset=None, eval_freq=None):
        train_dataset: tf.data.Dataset = train_dataset[0]
        train_iter = train_dataset.__iter__()

        if eval_dataset and not eval_freq:
            eval_freq = save_freq

        logging.info("Beginning training loop")
        last_time = start_time = time()
        for step in tf.range(self._optimizer.iterations + 1, tf.constant(train_steps + 1)):
            entry = next(train_iter)
            if len(entry) == 2:
                x_image = entry[0]
            else:
                x_image = next(train_iter)[2]
            x_image = tf.image.rgb_to_grayscale(x_image / 255.0) if self._grayscale else x_image / 255.0
            loss, reconstruction_loss, kl_loss = self.train_step(x_image)

            if step and step % print_freq == 0:
                curr_time = time()
                logging.info("Step: %6d | Loss: %6f | Recons. Loss: %6f | KL Loss: %6f | LR: %.5f | time/step: %.4f | Total Time: %7d",
                             step, loss, reconstruction_loss, kl_loss, self._optimizer._decayed_lr('float32'),
                             (curr_time-last_time)/print_freq, curr_time-start_time)
                last_time = curr_time
                with self._writer.as_default():
                    self._write_summaries(step, {"cost": loss, "reconstruction_cost": reconstruction_loss, "kl_cost": kl_loss})

            if step and step % save_freq == 0:
                self._ckpt_manager.save(step)

            if eval_dataset and step and step % eval_freq == 0:
                eval_start_time = time()
                eval_cost, eval_reconstruction, eval_kl = self.evaluate(eval_dataset)

                with self._writer.as_default():
                    self._write_summaries(step, {"eval_cost": eval_cost,
                                                 "eval_reconstruction": eval_reconstruction, "eval_kl": eval_kl})
                last_time = time()
                logging.info("Eval Done   | Loss: %.5f | Recons. Loss: %6f | KL Loss: %6f | Eval Time: %.4f | Total Time: %.0f",
                             eval_cost, eval_reconstruction, eval_kl, last_time-eval_start_time, last_time-start_time)

    def evaluate(self, eval_dataset):
        eval_dataset, _ = eval_dataset
        total_cost_mean, reconstruction_mean, kl_mean = tf.keras.metrics.Mean(), tf.keras.metrics.Mean(), tf.keras.metrics.Mean()

        for entry in eval_dataset.__iter__():
            if len(entry) == 2:
                x_image = entry[0]
            else:
                x_image = next(entry)[2]
            x_image = tf.image.rgb_to_grayscale(x_image/255.0) if self._grayscale else x_image / 255.0
            if x_image.shape[0] != self._png_dim:
                x_image = tf.image.resize(x_image, (self._png_dim, self._png_dim))

            outputs = self.forward(x_image, training=True)
            total_cost, reconstruction_cost, kl_cost = self.compute_loss(outputs, x_image)

            total_cost_mean(total_cost)
            reconstruction_mean(reconstruction_cost)
            kl_mean(kl_cost)

        return total_cost_mean.result(), reconstruction_mean.result(), kl_mean.result()

    def test(self, test_dataset, result_name, steps=None):
        self._sampling_dir = os.path.join(self._sampling_dir, result_name)
        test_dataset, _ = test_dataset

        # Begin Writing Child-Process
        process, write_queue = process_write_out(parallel_writer_vae_latent, (self._sampling_dir,))

        try:
            for step, entry in enumerate(test_dataset):
                if steps == step:
                    break

                x_image, class_names = entry[-2:]
                x_image = tf.image.rgb_to_grayscale(x_image / 255.0) if self._grayscale else x_image / 255.0
                if x_image.shape[0] != self._png_dim:
                    x_image = tf.image.resize(x_image, (self._png_dim, self._png_dim))
                reconstruction, [z, _, _] = self.forward(x_image, training=False)
                np_images, np_recons, np_z, np_class_names = x_image.numpy(), reconstruction.numpy(), z.numpy(), class_names.numpy()

                for idx in range(0, x_image.shape[0], 50):
                    write_queue.put({"rasterized_images": np_images[idx],
                                     "reconstructed_images": np_recons[idx],
                                     "latent_embedding": np_z[idx],
                                     "class_names": np_class_names[idx]})
            write_queue.put(None)
        except:
            process.terminate()
            raise

        process.join()

    @tf.function
    def train_step(self, x_image):
        if x_image.shape[0] != self._png_dim:
            x_image = tf.image.resize(x_image, (self._png_dim, self._png_dim))

        with tf.GradientTape() as tape:
            outputs = self.forward(x_image, training=True)
            total_loss, reconstruction_loss, kl_loss = self.compute_loss(outputs, x_image)

        grads = tape.gradient(total_loss,
                              self._conv_encoder.trainable_variables +
                              self._mu.trainable_variables + self._var.trainable_variables +
                            self._conv_decoder.trainable_variables)
        self._optimizer.apply_gradients(zip(grads,
                                            self._conv_encoder.trainable_variables +
                                            self._mu.trainable_variables + self._var.trainable_variables +
                                            self._conv_decoder.trainable_variables))

        return total_loss, reconstruction_loss, kl_loss

    @tf.function
    def compute_loss(self, outputs, ground_truth):
        reconstruction, params = outputs
        z, mean, logvar = params

        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=reconstruction, labels=ground_truth)
        reconstruction_loss_batched = tf.reduce_mean(cross_ent, axis=[1, 2, 3])
        kl_loss_batched = -0.5 * tf.reduce_mean((1 + logvar - tf.square(mean) - tf.exp(logvar)))
        kl_loss_batched = tf.maximum(kl_loss_batched, self._kl_tolerance) * self._kl_weight

        total_loss_batched = reconstruction_loss_batched + kl_loss_batched
        # total_loss_batched = reconstruction_loss_batched

        reconstruction_loss = tf.reduce_mean(reconstruction_loss_batched)
        kl_loss = tf.reduce_mean(kl_loss_batched)
        total_loss = tf.reduce_mean(total_loss_batched)

        return total_loss, reconstruction_loss, kl_loss

    @tf.function
    def embed(self, x_image, training=False):
        if x_image.shape[0] != self._png_dim:
            x_image = tf.image.resize(x_image, (self._png_dim, self._png_dim))
        if tf.reduce_max(x_image) > 1.0:
            x_image = x_image / 255.0

        x = self._conv_encoder(x_image, training)

        mu, var = self._mu(x, training=training), self._var(x, training=training)

        # Re-parameterize
        eps = tf.random.normal(shape=mu.shape)
        z = mu + eps * var
        return z, mu, var

    @tf.function
    def forward(self, x_image, training):
        if x_image.shape[1] != self._png_dim:
            x_image = tf.image.resize(x_image, (self._png_dim, self._png_dim))
        if tf.reduce_max(x_image) > 1.0:
            x_image = x_image / 255.0

        x = self._conv_encoder(x_image, training)

        mu, var = self._mu(x, training=training), self._var(x, training=training)
        # Re-parameterize
        eps = tf.random.normal(shape=mu.shape)
        z = mu + eps * var

        x = tf.reshape(z, (z.shape[0], 1, 1, z.shape[-1]))
        x = self._conv_decoder(x, training)

        if not training:
            x = tf.keras.activations.sigmoid(x)

        return x, (z, mu, var)

    @tf.function
    def decode(self, z, training):
        x = tf.reshape(z, (z.shape[0], 1, 1, z.shape[-1]))
        x = self._conv_decoder(x, training)

        if not training:
            x = tf.keras.activations.sigmoid(x)

        return x
