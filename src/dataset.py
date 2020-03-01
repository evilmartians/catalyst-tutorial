import cv2
import numpy as np
import torch

from torch.utils.data import Dataset

CLASSES = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}


class MNISTDataset(Dataset):
    """ dataset to read mnist png data, apply transforms if there any,
        and return with corresponded class taken from image path

        args:
        paths: paths of images
        mode: mode of dataset usage to understand if targets
            need to be extracted or not
        transforms (a.compose): data transformation pipeline
            from albumentations package (e.g. flip, scale, etc.)
    """

    def __init__(self, paths: list, mode: str, transforms=None):
        self.paths = paths
        mode_err = "Mode should be one of `train`, `valid` or `infer`"
        assert mode in ["train", "valid", "infer"], mode_err
        self.mode = mode
        self.transforms = transforms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # We need to cast Path instance to str
        # as cv2.imread is waiting for a string file path
        item = {"paths": str(self.paths[idx])}
        img = cv2.imread(item["paths"])
        if self.transforms is not None:
            img = self.transforms(image=img)["image"]
        img = np.moveaxis(img, -1, 0)
        item["features"] = torch.from_numpy(img)

        if self.mode != "infer":
            # We need to provide a numerical index of a class, not string,
            # so we cast it to int
            item["targets"] = int(item["paths"].split("/")[-2])

        return item
