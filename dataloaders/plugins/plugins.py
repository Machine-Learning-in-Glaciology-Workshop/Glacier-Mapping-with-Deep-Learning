import abc
import dataloaders.transformations
import numpy as np


class Plugin(abc.ABC):
    def set_dataloader(self, dataloader):
        self.dataloader = dataloader

    def get_sampler(self):
        if not self.dataloader:
            raise ValueError()
        return self.dataloader.sampler

    def __init_subclass__(cls, **kwargs):
        cls.has_before_indexing_behaviour = not (cls.before_indexing == Plugin.before_indexing)
        cls.has_after_indexing_behaviour = not (cls.after_indexing == Plugin.after_indexing)
        cls.has_on_sampling_behaviour = not (cls.on_sampling == Plugin.on_sampling)
        cls.has_on_finalising_behaviour = not (cls.on_finalising == Plugin.on_finalising)

    def before_indexing(self, sampler):
        pass

    def after_indexing(self, sampler):
        pass

    def on_sampling(self, sample):
        return sample

    def on_finalising(self, batch_x, batch_y):
        return batch_x, batch_y


class TileFilter(Plugin):
    def __init__(self, filters):
        self.filters = filters

    def before_indexing(self, sampler):
        dataset = sampler.dataset
        tiles = sampler.tiles
        filtered_tiles = []
        for tile in tiles:
            tile_group = dataset[tile]
            if self.apply_filters(tile_group):
                filtered_tiles.append(tile)
        sampler.tiles = filtered_tiles

    def apply_filters(self, tile_group):
        filter_outputs = [filter(tile_group) for filter in self.filters]
        return all(filter_outputs)


class Augmentation(Plugin):
    def __init__(self, transformations):
        self.transformations = transformations

    def on_sampling(self, sample):
        dataloaders.transformations.apply_transformations(
            sample, self.transformations
        )
        return sample


class AddDeepSupervision(Plugin):
    def __init__(self, n_branches=2):
        self.n_branches = n_branches

    def on_finalising(self, batch_x, batch_y):
        batch_y = [batch_y for _ in range(self.n_branches)]
        return batch_x, batch_y


class RepeatWithMandatoryTransformations(Plugin):
    def __init__(self, transformations):
        self.transformations = transformations

    def after_indexing(self, sampler):
        sampler.n_patches *= 2

    def on_sampling(self, sample):
        sample_copy = {key: value.copy() for key, value in sample.items()}
        dataloaders.transformations.apply_transformations(
            sample_copy, self.transformations
        )
        self.dataloader.batch_list.append(sample_copy)
        self.dataloader.sample_idx += 1
        return sample
