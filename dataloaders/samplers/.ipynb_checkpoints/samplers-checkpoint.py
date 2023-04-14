import numpy as np
import abc


class Sampler(abc.ABC):
    def __init__(self, dataset, patch_size):
        self.dataset = dataset
        self.tiles = list(dataset.keys())
        self.patch_size = patch_size

    def set_dataloader(self, dataloader):
        self.dataloader = dataloader

    def index(self):
        self.n_patches = 0 
        self.regions = set()
        for tile in self.tiles:
            attrs = self.dataset[tile].attrs
            height = attrs["height"] + 2 * attrs["pad_height"] + 1
            width = attrs["width"] + 2 * attrs["pad_width"] + 1
            self.n_patches += (height // self.patch_size) * (width // self.patch_size)

    def reset(self):
        pass

    @abc.abstractmethod
    def sample(self):
        raise NotImplementedError


class RandomSampler(Sampler):
    def __init__(self, dataset, patch_size, features=["features"], labels="outlines"):
        super(RandomSampler, self).__init__(dataset, patch_size)
        self.features = features
        self.labels = labels

    def sample(self):
        self.sample_image()
        self.sample_patch()
        return self.patch

    def sample_image(self):
        tile = np.random.choice(self.tiles)
        self.tile_group = self.dataset[tile]

    def sample_patch(self):
        height, width, _ = self.tile_group[self.features[0]].shape
        self.y = np.random.choice(height - self.patch_size)
        self.x = np.random.choice(width - self.patch_size)
        self.patch = {}
        for feature in self.features:
            feature_patch = self.tile_group[feature][
                self.y:self.y + self.patch_size, self.x:self.x + self.patch_size, :
            ]
            self.patch[feature] = feature_patch
        self.patch[self.labels] = self.tile_group[self.labels][
            self.y:self.y + self.patch_size, self.x:self.x + self.patch_size, :
        ].astype(np.double)


class ConsecutiveSampler(Sampler):
    def __init__(self, dataset, patch_size, features=["features"], labels="outlines"):
        super(ConsecutiveSampler, self).__init__(dataset, patch_size)
        self.features = features
        self.labels = labels
        self.reset()

    def reset(self):
        self.tile_idx = 0
        self.x = 0
        self.y = 0

    def sample(self):
        self.sample_image()
        height, width, _ = self.tile_group[self.features[0]].shape
        if self.x + self.patch_size > width:
            self.x = 0
            self.y += self.patch_size
        if self.y + self.patch_size > height:
            self.y = 0
            self.x = 0
            self.tile_idx += 1
            self.sample_image()
        self.patch = {}
        for feature in self.features:
            feature_patch = self.tile_group[feature][
                self.y:self.y + self.patch_size, self.x:self.x + self.patch_size, :
            ]
            self.patch[feature] = feature_patch
        self.patch[self.labels] = self.tile_group[self.labels][
            self.y:self.y + self.patch_size, self.x:self.x + self.patch_size, :
        ].astype(np.double)
        self.x += self.patch_size
        return self.patch

    def sample_image(self):
        if self.tile_idx >= len(self.tiles):
            self.reset()
        tile = self.tiles[self.tile_idx]
        self.tile_group = self.dataset[tile]
