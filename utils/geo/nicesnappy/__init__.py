import nicesnappy.utils as utils
from nicesnappy.operators import *


def initialize(snappy_dir):
    snappy = utils.get_snappy(snappy_dir)
    utils.set_snappy(snappy)


def initialize_with_module(snappy):
    utils.set_snappy(snappy)
