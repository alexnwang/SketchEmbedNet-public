import os
import traceback
import random
import tensorflow as tf
import numpy as np

from absl import logging

from .dataset_base import DatasetBase

class DatasetEpisodic(DatasetBase):
    def __init__(self, data_dir, params):
        super(DatasetEpisodic, self).__init__(data_dir, params)

    def _load_episodic_or_batch(self, data_path, repeat):
        if self._mode == "batch":
            # If not episdic, i.e. conventional loading
            files = []
            for alphabet in self._split.split(","):
                path = os.path.join(data_path, alphabet)
                if os.path.isdir(path):
                    file_list = os.listdir(os.path.join(data_path, alphabet))
                    files.extend([os.path.join(data_path, alphabet, c) for c in file_list])
                else:
                    files.append(path)

            return self._create_dataset_from_filepaths(files, repeat)
        elif self._mode == "episodic":
            episodes = []

            for episode_string in self._split.split(";"):
                files = []
                for collection in episode_string.split(","):
                    path = os.path.join(data_path, collection)
                    if os.path.isdir(path):
                        file_list = os.listdir(os.path.join(data_path, collection))
                        files.extend([os.path.join(data_path, collection, file) for file in file_list])
                    else:
                        files.append(path)
                episodes.append(",".join(files))

            return self._create_episodic_dataset_from_nested_filespaths(episodes, self.shot, self.way, repeat)
        else:
            logging.fatal("Dataset mode not \"episodic\" or \"batch\", value supplied: %s", self._mode)

    def _create_episodic_dataset_from_nested_filespaths(self, episodes, shot, way, repeat):
        try:
            npz = np.load(episodes[0].split(",")[0], allow_pickle=True, encoding='latin1')
            npz_collections = self._filter_collections(npz.files)

            # shapes = [npz[key][0].shape for key in npz_collections] * 2
            types = tuple([tf.as_dtype(npz[key].dtype) for key in npz_collections] * 2)
        except Exception as e:
            logging.error("%s file load unsuccessful from %s \n %s", type(self).__name__, episodes, str(e))
            logging.info(traceback.format_exc())
            raise e
        dataset = tf.data.Dataset.from_generator(self._make_episode_generator,
                                                 args=(episodes, shot, way, npz_collections),
                                                 output_types=types)

        if self._augmentations:
            dataset = dataset.interleave(lambda *args: tf.data.Dataset.from_generator(self._apply_augmentations_generator,
                                                                                      args=args,
                                                                                      output_types=types),
                                         num_parallel_calls=self._num_parallel_calls,
                                         block_length=self._block_length,
                                         cycle_length=self._cycle_length)

        if self._shuffle:
            dataset = dataset.shuffle(self._buff_size * self._batch_size)
        if repeat:
            dataset = dataset.repeat()

        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset

    def _make_episode_generator(self, episodes, shot, way, npz_collections):
        for episode_classes_string in episodes:
            episode_classes_list = episode_classes_string.decode('utf-8').split(",")
            episode_classes = random.sample(list(episode_classes_list), way)

            support = [[] for _ in range(len(npz_collections))]
            query = [[] for _ in range(len(npz_collections))]
            for class_file in episode_classes:
                try:
                    npz = np.load(class_file, allow_pickle=True, encoding='latin1')
                except FileNotFoundError as error:
                    logging.fatal("Shard not found when producing generator fn: %s", class_file)
                    raise error

                collections = [npz[key.decode('utf-8')] for key in npz_collections]

                sample_idxs = np.linspace(0., float(len(collections[0])-1), len(collections[0])).astype(np.int32)
                np.random.shuffle(sample_idxs)

                for idx, collection in enumerate(collections):
                    support[idx].append(collection[sample_idxs[:shot]])

                for idx, collection in enumerate(collections):
                    query[idx].append(collection[sample_idxs[shot:]])

            yield tuple([np.concatenate(x, axis=0) for x in support] + [np.concatenate(x, axis=0) for x in query])
