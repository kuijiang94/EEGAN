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
from TESTGAN import Model

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.reset_default_graph()


is_training = tf.placeholder(tf.bool, [])
# coord = tf.train.Coordinator()
# threads = tf.train.start_queue_runners(sess=sess, coord=coord)
# try:
	# print(sess.run(tf.global_variables_initializer()))
# except Exception as e:
		# #Report exceptions to the coordinator
	# coord.request_stop(e)


img_path = '..\\test\\test30\\'
file = os.listdir(img_path)
save_path = '..\\test\\result30\\'

if not os.path.exists(save_path):
    os.mkdir(save_path)

st_time=time.time()
num =1
for f in file:
    pic_path = os.path.join(img_path, f)
    file_name = f
    img = cv2.imread(pic_path)
    W, H = img.shape[:2]
    print(img.shape)
    img = img / 127.5 - 1
    input_ = np.zeros((1, W, H, 3))#16
    print(input_.shape)
    input_[0] = img
    if num==1:
        W_1 = W
        H_1 = H
        x = tf.placeholder(tf.float32, [None, W, H, 3])
        model = Model(x, is_training, 1)#16
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        saver = tf.train.Saver()
        saver.restore(sess, '../EEGANx4/epoch45')#93
        fake, mos, frame = sess.run(
            [model.imitation_sr, model.base_sr, model.frame_sr],
            feed_dict={x: input_, is_training: False})
    else:
        if W_1==W and H_1 == H:
            fake = sess.run(
                [model.ZConv_VDSR],
                feed_dict={x: input_, is_training: False})
        else:
            #sess.close()
            W_1 = W
            H_1 = H
            x = tf.placeholder(tf.float32, [None, W, H, 3])
            model = Model(x, is_training, 1)#16
            init = tf.global_variables_initializer()
            sess = tf.Session()
            sess.run(init)
            saver = tf.train.Saver()
            saver.restore(sess, '../EEGANx4/epoch45')#93
            fake, mos, frame = sess.run(
                [model.imitation_sr, model.base_sr, model.frame_sr],
                feed_dict={x: input_, is_training: False})
    img = fake[0]

    im = np.uint8(np.clip((img[0]+1)*127.5,0,255.0))
    cv2.imwrite(os.path.join(save_path, file_name), im)

    num+=1
ed_time=time.time()
cost_time=ed_time-st_time
print('spent {} s.'.format(cost_time))