import numpy as np


def get_z_factor(lat):
    LATS = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80])
    ZS = np.array([
        0.00000898, 0.00000912, 0.00000956, 0.00001036, 0.00001171,
        0.00001395, 0.00001792, 0.00002619, 0.00005156
    ])
    return np.interp(lat, LATS, ZS)


def are_bounds_intersect(bounds1, bounds2):
        xmin1, ymin1, xmax1, ymax1 = bounds1
        xmin2, ymin2, xmax2, ymax2 = bounds2

        if xmax1 < xmin2:
            return False
        if xmax2 < xmin1:
            return False
        if ymax1 < ymin2:
            return False
        if ymax2 < ymin1:
            return False
        
        return True
    