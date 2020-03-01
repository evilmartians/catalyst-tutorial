import random
from pathlib import Path
import albumentations as A
from collections import OrderedDict
from catalyst.dl import ConfigExperiment
from .dataset import MNISTDataset


class Experiment(ConfigExperiment):

    @staticmethod
    def get_transforms():
        return A.Normalize()

    def get_datasets(self, stage: str, **kwargs):
        datasets = OrderedDict()
        data_params = self.stages_config[stage]["data_params"]

        if stage != "infer":
            train_path = Path(data_params["train_dir"])

            imgs = list(train_path.glob('**/*.png'))
            split_on = int(len(imgs) * data_params["valid_size"])
            random.shuffle(imgs)
            train_imgs, valid_imgs = imgs[split_on:], imgs[:split_on]

            datasets["train"] = MNISTDataset(paths=train_imgs,
                                             mode="train",
                                             transforms=self.get_transforms())

            datasets["valid"] = MNISTDataset(paths=valid_imgs,
                                             mode="valid",
                                             transforms=self.get_transforms())
        else:
            test_path = Path(data_params["test_dir"])
            imgs = list(test_path.glob('**/*.png'))
            datasets["infer"] = MNISTDataset(paths=imgs,
                                             mode="infer",
                                             transforms=self.get_transforms())

        return datasets
