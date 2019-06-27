import numpy as np 
def wrap_around(x):
    # x \in [-pi,pi)
    if x >= np.pi:
        return x - 2*np.pi
    elif x < -np.pi:
        return x + 2*np.pi
    else:
        return x

def relative_measure(x_target, x_main):
    diff = x_target[:2] - x_main[:2]
    r = np.sqrt(np.sum(diff**2))
    alpha = wrap_around(np.arctan2(diff[1],diff[0]) - x_main[2])
    return r, alpha, diff


def coord_change2b(vec, ang):
    assert(len(vec) == 2)
    # R^T * v
    return np.matmul([[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]], vec)

