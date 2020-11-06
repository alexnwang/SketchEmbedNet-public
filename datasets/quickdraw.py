import os
from concurrent.futures.process import ProcessPoolExecutor
from itertools import repeat
from math import ceil
from time import time

import numpy as np
import psutil as psutil
from absl import logging

from datasets.base import register_dataset, DatasetBase
from util import get_normalizing_scale_factor, quickdraw_process, ST2_classes, ST1_classes


@register_dataset("quickdraw")
class Quickdraw(DatasetBase):
    def __init__(self, data_dir, params):
        super(Quickdraw, self).__init__(data_dir, params)

        self._dataset_path = os.path.join(self._data_dir, 'quickdraw')

    def load(self, repeat=True):
        data_path = os.path.join(self._dataset_path, 'caches', self._split)
        files = [os.path.join(data_path, shard_name) for shard_name in os.listdir(data_path)]

        return self._create_dataset_from_filepaths(files, repeat)

    def _filter_collections(self, files):
        """
        Selects files from archive.
        :param files:
        :return: y_strokes(ground_truth), y_strokes(teacher), x_image, class_name
        """
        files = sorted(files)
        return files[2], files[2], files[1], files[0]

    def prepare(self, FLAGS, max_seq_len=64, shard_size=1000, png_dims=(28, 28), unit_var=True, classes=ST1_classes):
        """
        Parallelized processing function for the Quickdraw dataset to convert .npz files into files for model ingestion.
        :param FLAGS:
        :param max_seq_len:
        :param shard_size:
        :param png_dims:
        :param unit_var:
        :param classes:
        :return:
        """
        save_dir = os.path.join(self._dataset_path, "caches", self._split)
        raw_dir = os.path.join(self._dataset_path, "raw")

        os.makedirs(save_dir, exist_ok=True)
        files = [os.path.join(raw_dir, file_path+".npz") for file_path in classes]
        logging.info('Loading NPZ Files | Num Classes: %d', len(files))
        all_sketches = np.empty((0, 2))
        for file in files:
            try:
                npz = np.load(file, encoding='latin1', allow_pickle=True)
            except IOError:
                logging.error("Numpy unable to load dataset file: {}".format(file))
                continue

            class_name = np.array([file.split('/')[-1].split('.')[0]])

            sketches = np.reshape(npz['train'], (-1, 1))

            classes = np.tile(np.reshape(class_name, (1, -1)), sketches.shape)

            data = np.concatenate((sketches, classes), axis=1)

            if max_seq_len:
                bool_array = np.array([sketch.shape[0] <= max_seq_len for sketch in data[:, 0]])
                data = data[bool_array]

            all_sketches = np.concatenate((all_sketches, data))
            logging.info("Loaded npz: %s | Taking samples %d/%d | Total samples: %d",
                         file,  data.shape[0], sketches.shape[0], all_sketches.shape[0])

        if not max_seq_len:
            max_seq_len = max([len(x) for x in all_sketches[:, 0]])

        # Scale all offsets to be of unit variance (makes image very small)
        if unit_var:
            normalizing_scale_factor = get_normalizing_scale_factor(all_sketches[:, 0:1])
        else:
            normalizing_scale_factor = 1.0

        # Randomize
        np.random.shuffle(all_sketches)
        logging.info("Beginning Processing | %s sketches | %s classes | Max Sequence Length: %s",
                     all_sketches.shape[0], len(files), max_seq_len)

        cpu_count = psutil.cpu_count(logical=False)
        workers_per_cpu = 2
        with ProcessPoolExecutor(max_workers=cpu_count * workers_per_cpu) as executor:
            out = executor.map(quickdraw_process,
                               (all_sketches[i: i + shard_size] for i in range(0, all_sketches.shape[0], shard_size)),
                               repeat(max_seq_len),
                               repeat(png_dims),
                               (os.path.join(save_dir, "{}.npz".format(i)) for i in range(ceil(all_sketches.shape[0] // shard_size))),
                               repeat(normalizing_scale_factor),
                               chunksize=1)

            batch_count = 0
            last_time = time()
            try:
                for write_signal in out:
                    batch_count += write_signal
                    if batch_count % cpu_count == 0:
                        curr_time = time()
                        logging.info("Processed batch: {:5} | Total: {:8}/{:8} | Time/Batch: {:8.2f} | Time/Image: {:8.8f}"
                                    .format(batch_count,
                                            min(batch_count * shard_size, all_sketches.shape[0]),
                                            all_sketches.shape[0],
                                            (curr_time - last_time) / cpu_count,
                                            (curr_time - last_time) / (cpu_count * shard_size)))
                        last_time = curr_time
            except StopIteration:
                logging.info("Processing Done")
