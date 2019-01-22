import os
import glob
import shutil
import cv2
#import dlib
import numpy as np
from tqdm import tqdm
import requests
import tarfile

# dlib
#detector = dlib.get_frontal_face_detector()

def download():
    print('... downloading')
    url = 'http://vis-www.cs.umass.edu/lfw/lfw.tgz'
    path = 'data/lfw.tgz'
    
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


'''def detect_face(img):
    h, w = img.shape[:2]
    dets = img#detector(img, 1)
    if dets is None or len(dets) != 1:
        return None
    d = dets[0]
    if d.left() < 0 or d.top() < 0 or d.right() > w or d.bottom() > h:
        return None
    face = img[d.top():d.bottom(), d.left():d.right()]
    face = cv2.resize(face, (96, 96))
    return face'''


def preprocess():
    print('... loading data')
    os.mkdir('data/raw')
    os.mkdir('data/raw/train')
    os.mkdir('data/raw/test')
    os.mkdir('data/npy')

    persons = glob.glob('data/lfw/*')
    paths = np.array(
        [e for x in [glob.glob(os.path.join(person, '*')) 
        for person in persons] for e in x])
    np.random.shuffle(paths)

    r = int(len(paths) * 0.99)
    train_paths = paths[:r]
    test_paths = paths[r:]

    x_train = []
    pbar = tqdm(total=(len(train_paths)))
    for i, d in enumerate(train_paths):
        pbar.update(1)
        img = cv2.imread(d)
        face = img#detect_face(img)
        face = cv2.resize(face, (96, 96))#128
        if face is None:
            continue
        x_train.append(face)
        name = "{}.png".format("{0:05d}".format(i))
        imgpath = os.path.join('data/raw/train', name)
        cv2.imwrite(imgpath, face)
    pbar.close()

    x_test = []
    pbar = tqdm(total=(len(test_paths)))
    for i, d in enumerate(test_paths):
        pbar.update(1)
        img = cv2.imread(d)
        face = img#detect_face(img)
        face = cv2.resize(face, (96, 96))#128
        if face is None:
            continue
        x_test.append(face)
        name = "{}.png".format("{0:05d}".format(i))
        imgpath = os.path.join('data/raw/test', name)
        cv2.imwrite(imgpath, face)
    pbar.close()

    x_train = np.array(x_train, dtype=np.uint8)
    x_test = np.array(x_test, dtype=np.uint8)
    np.save('data/npy/x_train.npy', x_train)
    np.save('data/npy/x_test.npy', x_test)


def main():
    #os.mkdir('data')
    #download()
    preprocess()


if __name__ == '__main__':
    main()

