import numpy as np

def arraygen(function):
    """Turns 0-ary function into array-generator, given a shape."""
    thefun = np.vectorize(lambda x: function())
    return lambda shape: thefun(np.empty(shape))

def pick_highest(v):
    highest = -1.0
    highest_index = 0
    for i in xrange(v.shape[0]):
        if v[i,0] > highest:
            highest = v[i,0]
            highest_index = i

    return highest_index


def pick_letter(alphabet, output):
    return alphabet[pick_highest(output)]

def extremify(v):
    h = pick_highest(v)
    result = np.ones_like(v)
    result[h] = -1
    return result*-1
