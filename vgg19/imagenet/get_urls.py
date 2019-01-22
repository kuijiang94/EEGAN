import os
import shutil
from tqdm import tqdm
import requests
import tarfile
import re
import sqlite3

def download_urls():
    lists = [
        ['http://image-net.org/imagenet_data/urls/imagenet_fall11_urls.tgz',
         'data/urls/imagenet_fall11_urls.tgz',
         'data/urls/fall11.txt'],
        ['http://image-net.org/imagenet_data/urls/imagenet_winter11_urls.tgz',
         'data/urls/imagenet_winter11_urls.tgz',
         'winter11.txt'],
        ['http://image-net.org/imagenet_data/urls/imagenet_spring10_urls.tgz',
         'data/urls/imagenet_spring10_urls.tgz',
         'data/urls/spring10.txt'],
        ['http://image-net.org/imagenet_data/urls/imagenet_fall09_urls.tgz',
         'data/urls/imagenet_fall09_urls.tgz',
         'data/urls/fall09.txt']]

    for list_ in lists:
        url = list_[0]
        tar_path = list_[1]
        txt_path = list_[2]

        file_size = int(requests.head(url).headers["content-length"])
        r = requests.get(url, stream=True)
        pbar = tqdm(total=file_size, unit="b", unit_scale=True)
        with open(tar_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                f.write(chunk)
                pbar.update(len(chunk))
        pbar.close()
        
        tar = tarfile.open(tar_path, 'r')
        for item in tar:
            tar.extract(item, '.')
            shutil.move(item.name, txt_path)


def main():
    os.mkdir('data')
    os.mkdir('data/urls')
    download_urls()


if __name__ == '__main__':
    main()

