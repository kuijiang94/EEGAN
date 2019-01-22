import numpy as np
import cv2
import os
import glob
from tqdm import tqdm

def _load(src):
    paths = glob.glob(src)
    paths.sort()
    paths = paths[:100] # 100 classes
    x = None
    t = []
    for path in tqdm(paths):
        id_ = int(os.path.basename(path).split('.')[0])
        c = np.load(path)
        if c.size == 0:
            continue
        l = [id_ for _ in range(c.shape[0])]
        if x is None:
            x = c
        else:
            x = np.concatenate((x, c), 0)
        t += l
    t = np.array(t)
    return [x, t]

def load():
    x_train, t_train = _load('./imagenet/data/npy/train/*')
    x_test, t_test = _load('./imagenet/data/npy/test/*')
    return x_train, t_train, x_test, t_test

