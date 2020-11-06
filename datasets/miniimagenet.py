import os

import pickle as pkl
import numpy as np

from PIL import Image
from absl import logging

from datasets import register_dataset, DatasetBase, DatasetEpisodic


@register_dataset("miniimagenet")
class MiniImageNet(DatasetEpisodic):
    def __init__(self, data_dir, params):
        super(MiniImageNet, self).__init__(data_dir, params)

        self._mode = params.mode
        self.way = params.way
        self.shot = params.shot

        self._dataset_path = os.path.join(self._data_dir, 'miniimagenet')

    def load(self, repeat=True):
        data_path = os.path.join(self._dataset_path, 'caches')

        if not self._split:
            self._split = self._split = ','.join(sorted(os.listdir(data_path)))

        return self._load_episodic_or_batch(data_path, repeat)

    def _filter_collections(self, files):
        """

        :param files: {"image", "class_dict"}
        :return: ["image", "class_name"]
        """

        files_sorted = sorted(files)
        return files_sorted[1], files_sorted[0]

    def prepare(self, FLAGS, png_dims=(84, 84), padding=None):
        """
        Preparation function for miniImageNet
        :param FLAGS:
        :param png_dims:
        :param padding:
        :return:
        """
        padding = padding if padding else round(min(png_dims) / 10.0) * 2

        save_dir = os.path.join(self._dataset_path, "caches")
        raw_dir = os.path.join(self._dataset_path, "raw")
        os.makedirs(save_dir, exist_ok=True)

        logging.info("Processing MiniImageNet | png_dimensions: %s | padding: %s", png_dims, padding)
        total_count = 0
        for pkl_file in [os.path.join(raw_dir, file) for file in os.listdir(raw_dir)]:
            with open(pkl_file, 'rb') as file:
                pkl_dict = pkl.load(file, encoding='latin1')

            img_data, class_dict = pkl_dict["image_data"], pkl_dict["class_dict"]

            for imgnet_class_id in class_dict.keys():
                logging.info("Processing Imagenet ID: %s", imgnet_class_id)
                accumulate = {"image": [], "class_name": []}
                char_save_path = os.path.join(save_dir, imgnet_class_id + ".npz")

                per_class_count = 0
                for idx in class_dict[imgnet_class_id]:
                    image: Image.Image = Image.fromarray(img_data[idx])
                    image = image.resize(size=png_dims)

                    accumulate['image'].append(np.array(image, dtype=np.float32))
                    accumulate['class_name'].append(imgnet_class_id)
                    per_class_count += 1

                logging.info("Per Class Count: %s", per_class_count)
                total_count += per_class_count
                np.savez(char_save_path, **accumulate)
        logging.info("Processing done, total count: %s", total_count)
