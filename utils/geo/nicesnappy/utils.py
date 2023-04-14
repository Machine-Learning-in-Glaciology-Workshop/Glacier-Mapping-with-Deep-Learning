import sys


snap = None


def get_snappy(snappy_dir):
    sys.path.append(snappy_dir)
    import snappy
    return snappy


def set_snappy(snappy_module):
    global snap
    snap = snappy_module
