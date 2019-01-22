import numpy as np
import scipy
import cv2
import os
import glob
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
sys.path.append('../../utils')
sys.path.append('../../vgg19')
from TESTEEGANMULTI import SRGAN
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#image_size1 = 480
image_size1 = 204#720,204,512,456,320
image_size2 = 480#720,480,512,600,480
x = tf.placeholder(tf.float32, [None, image_size1, image_size2, 3])#96
is_training = tf.placeholder(tf.bool, [])

model = SRGAN(x, is_training, 1)#16
sess = tf.Session()
#st_time=time.time()
init = tf.global_variables_initializer()
sess.run(init)

saver = tf.train.Saver()
saver.restore(sess, '../EEGANx4/epoch45')#93

img_path = 'E:/jiangkui/shiyan/WX/EDGAN/src/test/test12/'#real10truth
file = os.listdir(img_path)
save_path = 'E:/jiangkui/shiyan/WX/EDGAN/src/test/result12/'

if not os.path.exists(save_path):
    os.mkdir(save_path)
for f in file:
    pic_path = os.path.join(img_path, f)
    file_name = f
    img = cv2.imread(pic_path)
    face = img
    face = face / 127.5 - 1
    print(face.shape)
    input_ = np.zeros((1, image_size1, image_size2, 3))#16
    print(input_.shape)
    input_[0] = face
    st_time=time.time()
    fake, mos, frame = sess.run(
        [model.imitation_sr, model.base_sr, model.frame_sr],
        feed_dict={x: input_, is_training: False})
    ed_time=time.time()
    cost_time=ed_time-st_time
    print('spent {} s.'.format(cost_time))
    img = fake[0]
    #im = np.uint8((img+1)*127.5)
    im = np.uint8(np.clip((img+1)*127.5,0,255.0))
    cv2.imwrite(os.path.join(save_path, file_name), im)