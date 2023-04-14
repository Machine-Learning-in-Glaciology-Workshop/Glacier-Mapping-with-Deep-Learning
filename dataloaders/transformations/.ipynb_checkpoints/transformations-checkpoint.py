import numpy as np
import cv2


def random_vertical_flip(p=0.5):
    def transform(patch):
        if np.random.random() > p:
            return
        for feature in patch:
            if len(patch[feature].shape) != 3:
                continue
            original = patch[feature]
            transformed = np.flip(original, axis=0)
            patch[feature] = transformed
    return transform


def random_horizontal_flip(p=0.5):
    def transform(patch):
        if np.random.random() > p:
            return
        for feature in patch:
            if len(patch[feature].shape) != 3:
                continue
            original = patch[feature]
            transformed = np.flip(original, axis=1)
            patch[feature] = transformed
    return transform


def random_rotation(p=0.75):
    def transform(patch):
        if np.random.random() > p:
            return
        k = np.random.choice([1, 2, 3])
        for feature in patch:
            if len(patch[feature].shape) != 3:
                continue
            original = patch[feature]
            transformed = np.rot90(original, k, axes=(1, 0))
            patch[feature] = transformed
    return transform


def crop_and_scale(patch_size, scale=0.8, p=0.5):
    def transform(patch):
        if np.random.random() > p:
            return
        scale_coef = np.random.random() * (1 - scale) + scale
        new_size = int(scale_coef * patch_size)
        y = np.random.choice(patch_size - new_size)
        x = np.random.choice(patch_size - new_size)
        for feature in patch:
            if len(patch[feature].shape) != 3:
                continue
            original = patch[feature]
            crop = original[y:y + new_size, x:x + new_size, :]
            _, _, depth = crop.shape
            transformed = np.empty((patch_size, patch_size, depth))
            for channel_idx in range(depth):
                transformed[:, :, channel_idx] = cv2.resize(
                    crop[:, :, channel_idx], (patch_size, patch_size)
                )
            patch[feature] = transformed
    return transform


def apply_transformations(patch, transformations):
    if not transformations:
        return 
    for transformation in transformations:
        transformation(patch)
