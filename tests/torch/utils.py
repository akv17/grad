import numpy as np


def compare_tensors(t0, t1):
    t0 = t0.astype(np.float32)
    t1 = t1.astype(np.float32)
    shape_flag = t0.shape == t1.shape
    eq_flag = np.allclose(t0, t1)
    flag = shape_flag and eq_flag
    return flag
