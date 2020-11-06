import os
from time import time

import tensorflow as tf
import numpy as np
from absl import logging

from .drawer import DrawerModel
from .vae import VAE
from models.base import TrainableModel, register_model
from util import stroke_three_format, scale_and_center_stroke_three, rasterize


@register_model('classifier')
class ClassifierModel(TrainableModel):
    def __init__(self, base_dir, model_id, params, training=True):
        """
        Initializes Resnet-101 classifier model.
        :param params:
        :param name:
        """

        if not params.class_list:
            logging.fatal("No categories defined, class_list is empty.")

        # ----- Model Parameters ----- #
        self._class_list = params.class_list.split(",")
        self._png_dims = params.png_dims
        self._num_classes = len(self._class_list)

        self._weights = params.weights

        # ----- Training Parameters ----- #
        self._lr = params.lr
        self._lr_schedule: dict = params.lr_schedule

        super(ClassifierModel, self).__init__(base_dir, model_id, training)

    def _build_model(self):
        self._class_lookup = tf.lookup.StaticVocabularyTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=self._class_list,
                values=tf.range(tf.size(self._class_list, out_type=tf.int64), dtype=tf.int64)),
            num_oov_buckets=1)

        resnet = tf.keras.applications.ResNet101(include_top=False, weights=self._weights, input_shape=(self._png_dims, self._png_dims, 3))
        self._model = tf.keras.Sequential([resnet,
                                           tf.keras.layers.Flatten(),
                                           tf.keras.layers.Dense(self._num_classes)])

        lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(list(self._lr_schedule.keys()),
                                                                  [self._lr] + list(self._lr_schedule.values()))
        self._optimizer = tf.optimizers.Adam(learning_rate=lr)

    def _checkpoint_model(self):
        ckpt = tf.train.Checkpoint(optimizer=self._optimizer,
                                   model=self._model)
        self._ckpt_manager = tf.train.CheckpointManager(ckpt, self._checkpoint_dir, max_to_keep=None)

        if self._ckpt_manager.latest_checkpoint:
            logging.info("Restoring Checkpoint: %s", self._ckpt_manager.latest_checkpoint)
            status = ckpt.restore(self._ckpt_manager.latest_checkpoint)
            if self.training:
                status.assert_existing_objects_matched()
            else:
                status.expect_partial()

    def train(self, train_dataset, train_steps, print_freq, save_freq,  eval_dataset=None, eval_freq=None):
        if eval_dataset and not eval_freq:
            eval_freq = save_freq

        train_dataset, _ = train_dataset  # The second element is a saveable file-based dataset, currently not used

        train_dataset = train_dataset.map(lambda a, b, image, class_str: (a, b,
                                                                          tf.image.resize(image, (self._png_dims, self._png_dims)),
                                                                          class_str))
        train_iter = train_dataset.__iter__()

        last_time = start_time = time()
        for step in tf.range(self._optimizer.iterations + 1, tf.constant(train_steps + 1)):
            x_image, class_name = next(train_iter)[2:4]
            class_name = tf.cast(tf.one_hot(self._class_lookup.lookup(class_name), depth=self._num_classes), dtype=tf.float32)

            loss, accuracy = self.train_step(x_image, class_name)

            if step and step % print_freq == 0:
                curr_time = time()
                logging.info("Step: %6d | Loss: %.5f | Accuracy: %.4f | LR: %.5f | time/step: %.4f | Total Time: %7d",
                             step, loss, accuracy, self._optimizer._decayed_lr('float32').numpy(),
                             (curr_time-last_time)/print_freq, curr_time-start_time)
                last_time = curr_time

                with self._writer.as_default():
                    self._write_summaries(step, {"lr": self._optimizer._decayed_lr('float32'),
                                                 "loss": loss, "accuracy": accuracy})

            if step and step % save_freq == 0:
                self._ckpt_manager.save(step)

            if eval_dataset and step and step % eval_freq == 0:
                self.evaluate(step, eval_dataset)

    def evaluate(self, step, eval_dataset):
        eval_dataset, _ = eval_dataset
        loss_mean, acc_mean = tf.keras.metrics.Mean(), tf.keras.metrics.Mean()

        eval_start_time = time()
        eval_dataset = eval_dataset.map(lambda a, b, image, class_str: (a, b,
                                                                        tf.image.resize(image, (self._png_dims, self._png_dims)),
                                                                        class_str), deterministic=False)
        for step, entry in enumerate(eval_dataset.__iter__()):
            x_image, class_name = entry[2:4]
            class_name = tf.cast(tf.one_hot(self._class_lookup.lookup(class_name), depth=self._num_classes), dtype=tf.float32)

            logits = self.forward(x_image, training=False)
            loss_batched = tf.nn.softmax_cross_entropy_with_logits(class_name, logits)
            loss = tf.reduce_mean(loss_batched)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(class_name, axis=1), tf.argmax(logits, axis=1)), dtype=tf.float32))

            loss_mean(loss)
            acc_mean(accuracy)
        last_time = time()

        eval_loss, eval_acc = loss_mean.result(), acc_mean.result()

        with self._writer.as_default():
            self._write_summaries(step, {"lr": self._optimizer._decayed_lr('float32'),
                                         "eval_loss": eval_loss, "eval_acc": eval_acc})
        logging.info("Eval Done   | Loss: %.5f | Accuracy: %.4f Eval Time: %.4f",
                     eval_loss, eval_acc, last_time - eval_start_time)

    def test(self, test_dataset, result_name, steps=None):
        logging.info("Beginning testing loop")
        sampling_dir = os.path.join(self._sampling_dir, result_name)
        os.makedirs(sampling_dir)
        write_file = open(os.path.join(sampling_dir, 'result_log.txt'), 'a')

        test_dataset, _ = test_dataset
        loss_mean, acc_mean = tf.keras.metrics.Mean(), tf.keras.metrics.Mean()

        for step, entry in enumerate(test_dataset):
            if step == steps:
                break

            x_image, class_name = entry[2:4]

            class_name = tf.cast(tf.one_hot(self._class_lookup.lookup(class_name), depth=self._num_classes), dtype=tf.float32)
            x_image = tf.cast(x_image, dtype=tf.float32)

            logits = self.forward(x_image, training=False)
            loss_batched = tf.nn.softmax_cross_entropy_with_logits(class_name, logits)
            loss = tf.reduce_mean(loss_batched)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(class_name, axis=1), tf.argmax(logits, axis=1)), dtype=tf.float32))

            loss_mean(loss)
            acc_mean(accuracy)

            logging.info('cumulative_loss: %s | cumulative_acc: %s', loss_mean.result(), acc_mean.result())
            write_file.write('cumulative_loss: {} | cumulative_acc: {} \n'.format(loss_mean.result(), acc_mean.result()))

    def classify_predictions(self, dataset, model, steps):
        """
        Samples examples from the model over the dataset.
        Sampled examples are then classified by the ResNet101 classifier.
        :param dataset:
        :param model:
        :param steps:
        :return:
        """
        dataset, _ = dataset
        loss_mean, acc_mean = tf.keras.metrics.Mean(), tf.keras.metrics.Mean()
        acc_list = []
        padding = round(self._png_dims / 10.) * 2

        for step, entry in enumerate(dataset):
            if step > steps:
                break

            x_image, class_name = entry[2:4]

            if isinstance(model, DrawerModel):
                input_strokes = tf.tile(tf.constant([[[0., 0., 1., 0., 0.]]]), (x_image.shape[0], 2, 1))
                predicted_strokes = model.forward(input_strokes, x_image, training=False, generation_length=64)[3]

                image_inputs = []
                for predicted_stroke in predicted_strokes.numpy():
                    stroke_three = stroke_three_format(predicted_stroke)
                    stroke_three_scaled_and_centered = scale_and_center_stroke_three(stroke_three, [self._png_dims] * 2, padding)
                    image_inputs.append(rasterize(stroke_three_scaled_and_centered, [self._png_dims] * 2))

                image_inputs = np.array(image_inputs, dtype=np.float32)
                image_inputs = tf.image.resize(image_inputs, (self._png_dims, self._png_dims))
            elif isinstance(model, VAE):
                image_inputs = model.forward(x_image, training=False)[0] * 255.0

            logits = self.forward(image_inputs, training=False)

            y_labels = tf.cast(tf.one_hot(self._class_lookup.lookup(class_name), depth=self._num_classes), dtype=tf.float32)

            loss_batched = tf.nn.softmax_cross_entropy_with_logits(y_labels, logits)
            loss = tf.reduce_mean(loss_batched)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_labels, axis=1), tf.argmax(logits, axis=1)), dtype=tf.float32))

            loss_mean(loss)
            acc_mean(accuracy)
            acc_list.append(accuracy)

            if step and step % 5 == 0:
                logging.info('cumulative_loss: %s | cumulative_acc: %s', loss_mean.result().numpy(), acc_mean.result().numpy())

        logging.info("Final Result | Mean Accuracy: %.4f | Std: %.4f | Var: %.4f | p95: %.4f",
                     acc_mean.result(), np.std(acc_list), np.var(acc_list), 1.96 * np.std(acc_list) / np.sqrt(len(acc_list)))

    @tf.function
    def forward(self, x_image, training):
        logits = self._model(x_image, training=True)
        return logits

    @tf.function
    def train_step(self, x_image, class_name):
        with tf.GradientTape() as tape:
            logits = self.forward(x_image, training=True)
            loss_batched = tf.nn.softmax_cross_entropy_with_logits(class_name, logits)

            loss = tf.reduce_mean(loss_batched)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(class_name, axis=1), tf.argmax(logits, axis=1)), dtype=tf.float32))

        grads = tape.gradient(loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))

        return loss, accuracy
