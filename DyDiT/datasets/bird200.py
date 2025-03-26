import os
import numpy as np
from torchvision.datasets.vision import VisionDataset
import PIL.Image


class Bird200(VisionDataset):

    def __init__(self, root,  transform=None):

        self.root = root
        self.transform = transform
        self.images_path = {}
        with open(os.path.join(self.root, 'images.txt')) as f:
            for line in f:
                image_id, path = line.split()
                self.images_path[image_id] = path

        self.class_ids = {}
        with open(os.path.join(self.root, 'image_class_labels.txt')) as f:
            for line in f:
                image_id, class_id = line.split()
                self.class_ids[image_id] = class_id
        
        self.data_id = []
        with open(os.path.join(self.root, 'train_test_split.txt')) as f:
            for line in f:
                image_id, is_train = line.split()
                self.data_id.append(image_id)
        # print("a")


    def __len__(self):
        return len(self.data_id)
    
    def __getitem__(self, index):
        """
        Args:
            index: index of training dataset
        Returns:
            image and its corresponding label
        """
        image_id = self.data_id[index]
        class_id = int(self._get_class_by_id(image_id)) - 1
        path = self._get_path_by_id(image_id)
        image = PIL.Image.open(os.path.join(self.root, 'images', path)).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, class_id

    def _get_path_by_id(self, image_id):

        return self.images_path[image_id]
    
    def _get_class_by_id(self, image_id):

        return self.class_ids[image_id]
