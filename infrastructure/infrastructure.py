import requests
import abc
import enum


class Satellites(enum.Enum):
    SENTINEL1 = enum.auto()
    SENTINEL2 = enum.auto()
    LANDSAT5 = enum.auto()
    LANDSAT7 = enum.auto()
    LANDSAT8 = enum.auto()
    LANDSAT9 = enum.auto()


class Sentinel1Products(enum.Enum):
    GRD = enum.auto()
    SLC = enum.auto()


class Sentinel2Products(enum.Enum):
    L1C = enum.auto()
    L2A = enum.auto()


class LandsatProducts(enum.Enum):
    L1G = enum.auto()
    L1T = enum.auto()
    L1GT = enum.auto()


class LandsatTiers(enum.Enum):
    T1 = enum.auto()
    T2 = enum.auto()


class EnvisatInstruments(enum.Enum):
    MERIS = enum.auto()
    ASAR = enum.auto()


class DataFinder(abc.ABC):
    def __init__(self, base_url):
        self.base_url = base_url

    def search_raw(self, **query_params):
        response = requests.get(url=self.base_url, params=query_params)
        data = response.json()
        return data

    @abc.abstractmethod
    def search(self, *args, start_date=None, end_date=None, geometry=None, max_cloud_cover=None, **kwargs):
        raise NotImplementedError
