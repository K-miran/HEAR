# The code is a changed version of https://github.com/noboevbo/nobos_torch_lib/blob/master/nobos_torch_lib/datasets/action_recognition_datasets/ehpi_dataset.py

import random
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose

image_size = [320, 240]

class FallData(Dataset):
    def __init__(self, inputs, label,transform):
        self.num_joints = 15
        self.x = inputs
        self.y = label
        self.transform = transform
        self.__length = len(self.y)

    def __len__(self):
        return self.__length

    def __getitem__(self, index):
        a = self.x[index]
        b = self.y[index]
        sample = {"x": self.x[index].copy(), "y": self.y[index]}
        sample = self.transform(sample)
        return sample


class OutsideIamge(object):
    def __init__(self):
        self.image_size = image_size

    def __call__(self, sample):
        ehpi_img = sample['x']
        tmp = np.copy(ehpi_img)
        ehpi_img[0, :, :][tmp[0, :, :] > self.image_size[0]] = 0
        ehpi_img[0, :, :][tmp[0, :, :] < 0] = 0
        ehpi_img[1, :, :][tmp[0, :, :] > self.image_size[0]] = 0
        ehpi_img[1, :, :][tmp[0, :, :] < 0] = 0
        ehpi_img[0, :, :][tmp[1, :, :] > self.image_size[0]] = 0
        ehpi_img[0, :, :][tmp[1, :, :] < 0] = 0
        ehpi_img[1, :, :][tmp[1, :, :] > self.image_size[0]] = 0
        ehpi_img[1, :, :][tmp[1, :, :] < 0] = 0
        sample['x'] = ehpi_img
        return sample


class Scale(object):
    def __init__(self):
        self.image_size = image_size

    def __call__(self, sample):
        ehpi_img = sample['x']
        tmp = np.copy(ehpi_img)
        curr_min_x = np.min(ehpi_img[0, :, :][ehpi_img[0, :, :] > 0])
        curr_min_y = np.min(ehpi_img[1, :, :][ehpi_img[1, :, :] > 0])
        curr_max_x = np.max(ehpi_img[0, :, :])
        curr_max_y = np.max(ehpi_img[1, :, :])
        max_factor_x = self.image_size[0] / curr_max_x
        max_factor_y = self.image_size[1] / curr_max_y
        min_factor_x = (self.image_size[0] * 0.1) / (curr_max_x - curr_min_x)
        min_factor_y = (self.image_size[1] * 0.1) / (curr_max_y - curr_min_y)
        min_factor = max(min_factor_x, min_factor_y)
        max_factor = min(max_factor_x, max_factor_y)
        factor = random.uniform(min_factor, max_factor)
        ehpi_img[0, :, :] = ehpi_img[0, :, :] * factor
        ehpi_img[1, :, :] = ehpi_img[1, :, :] * factor
        ehpi_img[0, :, :][tmp[0, :, :] == 0] = 0
        ehpi_img[1, :, :][tmp[1, :, :] == 0] = 0
        sample['x'] = ehpi_img
        return sample


class Translate(object):
    def __init__(self):
        self.image_size = image_size

    def __call__(self, sample):
        ehpi_img = sample['x']
        tmp = np.copy(ehpi_img)
        max_minus_translate_x = -np.min(ehpi_img[0, :, :][ehpi_img[0, :, :] > 0])
        max_minus_translate_y = -np.min(ehpi_img[1, :, :][ehpi_img[1, :, :] > 0])
        max_plus_translate_x = self.image_size[0] - np.max(ehpi_img[0, :, :])
        max_plus_translate_y = self.image_size[1] - np.max(ehpi_img[1, :, :])
        translate_x = random.uniform(max_minus_translate_x, max_plus_translate_x)
        translate_y = random.uniform(max_minus_translate_y, max_plus_translate_y)
        ehpi_img[0, :, :] = ehpi_img[0, :, :] + translate_x
        ehpi_img[1, :, :] = ehpi_img[1, :, :] + translate_y
        ehpi_img[0, :, :][tmp[0, :, :] == 0] = 0
        ehpi_img[1, :, :][tmp[1, :, :] == 0] = 0
        sample['x'] = ehpi_img
        return sample


class Normalize(object):
    def __call__(self, sample):
        ehpi_img = sample['x']
        tmp = np.copy(ehpi_img)
        curr_min_x = np.min(ehpi_img[0, :, :][ehpi_img[0, :, :] > 0])
        curr_min_y = np.min(ehpi_img[1, :, :][ehpi_img[1, :, :] > 0])
        ehpi_img[0, :, :] = ehpi_img[0, :, :] - curr_min_x
        ehpi_img[1, :, :] = ehpi_img[1, :, :] - curr_min_y
        max_factor_x = 1 / np.max(ehpi_img[0, :, :])
        max_factor_y = 1 / np.max(ehpi_img[1, :, :])
        ehpi_img[0, :, :] = ehpi_img[0, :, :] * max_factor_x
        ehpi_img[1, :, :] = ehpi_img[1, :, :] * max_factor_y
        ehpi_img[0, :, :][tmp[0, :, :] == 0] = 0
        ehpi_img[1, :, :][tmp[1, :, :] == 0] = 0
        sample['x'] = ehpi_img
        return sample


class Flip(object):
    def __init__(self, with_scores = True, left_indexes = [3, 4, 5, 9, 10, 11],
                 right_indexes = [6, 7, 8, 12, 13, 14]):
        self.step_size = 3 if with_scores else 2
        self.left_indexes = left_indexes
        self.right_indexes = right_indexes

    def __call__(self, sample):
        if bool(random.getrandbits(1)):
            return sample
        ehpi_img = sample['x']
        tmp = np.copy(ehpi_img)
        curr_min_x = np.min(ehpi_img[0, :, :][ehpi_img[0, :, :] > 0])
        curr_max_x = np.max(ehpi_img[0, :, :])

        reflect_x = (curr_max_x + curr_min_x) / 2
        ehpi_img[0, :, :] = reflect_x - (ehpi_img[0, :, :] - reflect_x)
        ehpi_img[0, :, :][tmp[0, :, :] == 0] = 0
        ehpi_img[1, :, :][tmp[1, :, :] == 0] = 0
        if bool(random.getrandbits(1)):
            for left_index, right_index in zip(self.left_indexes, self.right_indexes):
                tmp = np.copy(ehpi_img)
                ehpi_img[0:, :, left_index] = tmp[0:, :, right_index]
                ehpi_img[1:, :, left_index] = tmp[1:, :, right_index]
                ehpi_img[2:, :, left_index] = tmp[2:, :, right_index]
                ehpi_img[0:, :, right_index] = tmp[0:, :, left_index]
                ehpi_img[1:, :, right_index] = tmp[1:, :, left_index]
                ehpi_img[2:, :, right_index] = tmp[2:, :, left_index]
        sample['x'] = ehpi_img
        return sample


class RemoveJoints(object):
    def __init__(self, with_scores = True, indexes_to_remove = [11, 14],
                 indexes_to_remove_2 = [10, 13], probability = 0.25):
        self.step_size = 3 if with_scores else 2
        self.indexes_to_remove = indexes_to_remove
        self.indexes_to_remove_2 = indexes_to_remove_2
        self.probability = probability

    def __call__(self, sample):
        if not random.random() < self.probability:
            return sample
        ehpi_img = sample['x']
        for index in self.indexes_to_remove:
            ehpi_img[0:, :, index] = 0
            ehpi_img[1:, :, index] = 0
            ehpi_img[2:, :, index] = 0

        if random.random() < self.probability:
            for index in self.indexes_to_remove_2:
                ehpi_img[0:, :, index] = 0
                ehpi_img[1:, :, index] = 0
                ehpi_img[2:, :, index] = 0
        if ehpi_img.min() > 0: 
            sample['x'] = ehpi_img
        return sample
