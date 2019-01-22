import os
import shutil
import pickle
import numpy as np
import cv2
from tqdm import tqdm
import requests
import tarfile

def download():
    print('... downloading')
    url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    path = 'data/cifar-100-python.tar.gz'

    file_size = int(requests.head(url).headers['content-length'])
    r = requests.get(url, stream=True)
    pbar = tqdm(total=file_size, unit='b', unit_scale=True)
    with open(path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            f.write(chunk)
            pbar.update(len(chunk))
    pbar.close()

    tar = tarfile.open(path, 'r')
    for item in tar:
        tar.extract(item, 'data')


def preprocess():
    def make_data(data, label, mode):
        dir_ = os.path.join('data', 'raw', mode)
        os.mkdir(dir_)
        x = []; t = []
        for i, (d, l) in enumerate(zip(data, label)):
            bgr = np.array(d).reshape(3, 32, 32).transpose(1, 2, 0)
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            x.append(img)
            t.append(l)
            name = "{}_{}.jpg".format("{0:02d}".format(l), "{0:05d}".format(i))
            imgpath = os.path.join(dir_, name)
            cv2.imwrite(imgpath, img)
        x = np.array(x, dtype=np.uint8)
        t = np.array(t, dtype=np.int32)
        np.save('data/npy/x_{}.npy'.format(mode), x)
        np.save('data/npy/t_{}.npy'.format(mode), t)

    print('... loading data')
    os.mkdir('data/raw')
    os.mkdir('data/npy')

    with open('data/cifar-100-python/train', 'rb') as f:
        train = pickle.load(f, encoding='latin-1')
    make_data(train['data'], train['fine_labels'], 'train')

    with open('data/cifar-100-python/test', 'rb') as f:
        test = pickle.load(f, encoding='latin-1')
    make_data(test['data'], test['fine_labels'], 'test')

    with open('data/cifar-100-python/meta', 'rb') as f:
        meta = pickle.load(f)
    for name in meta['fine_label_names']:
        with open('data/class.txt', 'a') as f:
            f.write(name)
            f.write('\n')


def main():
    os.mkdir('data')
    download()
    preprocess()


if __name__ == '__main__':
    main()

