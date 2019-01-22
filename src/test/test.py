import numpy as np
import scipy.misc
import cv2
import os
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
sys.path.append('../../utils')
sys.path.append('../../vgg19')
from TESTGAN import SRGAN
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
image_size = 720#720,204,512
x = tf.placeholder(tf.float32, [1, image_size, image_size, 3])#96
is_training = tf.placeholder(tf.bool, [])

model = SRGAN(x, is_training, 1)#16
sess = tf.Session()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
try:
	print(sess.run(tf.global_variables_initializer()))
except Exception as e:
		#Report exceptions to the coordinator
	coord.request_stop(e)
#st_time=time.time()
init = tf.global_variables_initializer()
sess.run(init)

saver = tf.train.Saver()
saver.restore(sess, '../EEGANx4/epoch45')#93

img_path = '..\\test\\test30\\'
file = os.listdir(img_path)
save_path = '..\\test\\result30\\'
if not os.path.exists(save_path):
    os.mkdir(save_path)
st_time=time.time()
for f in file:
    pic_path = os.path.join(img_path, f)
    file_name = f
    img = cv2.imread(pic_path)
    #w, h = img.shape[:2]
    #print(img.shape)
    face = cv2.resize(img, (image_size, image_size))
    face = img
    face = face / 127.5 - 1
    #face = face / 255.0
    input_ = np.zeros((1, image_size, image_size, 3))#16
    print(input_.shape)
    input_[0] = face
    fake, mos, frame = sess.run(
        [model.imitation_sr, model.base_sr, model.frame_sr],
        feed_dict={x: input_, is_training: False})
    img = fake[0]
    #im = np.uint8((img+1)*127.5)
    im = np.uint8(np.clip((img+1)*127.5,0,255.0))
    cv2.imwrite(os.path.join(save_path, file_name), im)
ed_time=time.time()
cost_time=ed_time-st_time
print('spent {} s.'.format(cost_time))
