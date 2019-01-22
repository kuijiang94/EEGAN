import os
import glob
import cv2
from PIL import Image
import io
import numpy as np

def preprocess():
    pp = glob.glob('data/raw/*')
    pp.sort()
    for i, p in enumerate(pp):
        print(i, p)
        paths = glob.glob(os.path.join(p, '*'))
        x = []
        for path in paths:
            with open(path, 'rb') as img_bin:
                buff = io.BytesIO()
                buff.write(img_bin.read())
                buff.seek(0)
                try:
                    temp = np.array(Image.open(buff), dtype=np.uint8)
                except:
                    continue
                if temp.ndim != 3:
                    continue
                try:
                    img = cv2.cvtColor(temp, cv2.COLOR_RGB2BGR)
                except:
                    continue
            if img is None:
                continue
            img = cv2.resize(img, (96, 96))
            x.append(img)
        x = np.array(x, dtype=np.uint8)
        np.random.shuffle(x)
        r = int(len(x) * 0.95)
        x_train = x[:r]
        x_test = x[r:]
        print(x_train.shape, x_test.shape)
        id_ = "{0:04d}".format(i)
        np.save('data/npy/train/{}.npy'.format(id_), x_train)
        np.save('data/npy/test/{}.npy'.format(id_), x_test)


def main():
    os.mkdir('data/npy')
    os.mkdir('data/npy/train')
    os.mkdir('data/npy/test')
    preprocess()


if __name__ == '__main__':
    main()

