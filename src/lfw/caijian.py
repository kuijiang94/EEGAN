import os
from PIL import Image
from PIL import ImageChops
import numpy as np
import math
#import cv2

OFF = 1
SCALE = 4
X=-1

if __name__ == '__main__':
    data_path = '..\\lfw\\train\\'
    #file = os.listdir(data_path)
    save_path = '..\\lfw\\train\\'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    count =0
    num =0
    files = [
        os.path.join(data_path, filename)
        for filename in os.listdir(data_path)
        if 'png' in filename or 'tif' in filename]
    for filename in files:
        pic_path = os.path.join(data_path, filename)
        img = Image.open(pic_path)
        img = img.crop([0,0,img.size[0]-img.size[0]%100,img.size[1]-img.size[1]%100])
        #img = img.resize(f)
        #img = np.asarray(img)
        #if count%5==0:
        for i in range(0,img.size[0]-100,96):
            for j in range(0,img.size[1]-100,96):
                IMG = img.crop([i,j,i+96,j+96])
                IMG.save(os.path.join(save_path,'{}_{}_{}.png'.format(num,i//96,j//96)))
                num+=1

        count+=1
        print(count)
