import tensorflow as tf
import numpy as np
import sys
sys.path.append('../utils')
sys.path.append('../vgg19')
from layer import *
from vgg19 import VGG19
from ps import _PS
import random
        
class Model:
    def __init__(self, x, is_training, batch_size):
        self.batch_size = batch_size
        n,w,h,c = input.get_shape().as_list()
        self.weight = w//4
        self.height = h//4
        self.downscaled = self.downscale(x)
        self.bic_ref = tf.image.resize_images(self.downscaled, [self.weight*4, self.height*4], method=2)
        self.frame_sr, self.base_sr, self.imitation_sr  = self.generator(self.downscaled, is_training, False)
    
    def generator(self, x, is_training, reuse):
        with tf.variable_scope('generator', reuse=reuse):
            input = x
            with tf.variable_scope('conv1'):
                x = deconv_layer(
                    input, [3, 3, 128, 3], [self.batch_size, self.weight, self.height, 128], 1)
                x = lrelu(x)
            with tf.variable_scope('conv2'):
                x = deconv_layer(
                    x, [3, 3, 64, 128], [self.batch_size, self.weight, self.height, 64], 1)
                x = lrelu(x)
            #shortcut = x
            for i in range(6):
                with tf.variable_scope('block{}ex1'.format(i+1)):
                    x1=x2=x3=x
                    for j in range(3):
                        with tf.variable_scope('block{}_{}ex1'.format(i+1,j+1)):
                            with tf.variable_scope('ud1'):
                                a1 = lrelu(deconv_layer(x1, [3, 3, 64, 64], [self.batch_size, self.weight, self.height, 64], 1))
                                #a1 = batch_normalize(a1, is_training)
                            with tf.variable_scope('ud2'):
                                b1 = lrelu(deconv_layer(x2, [3, 3, 64, 64], [self.batch_size, self.weight, self.height, 64], 1))
                                #b1 = batch_normalize(b1, is_training)
                            with tf.variable_scope('ud3'):
                                c1 = lrelu(deconv_layer(x3, [3, 3, 64, 64], [self.batch_size, self.weight, self.height, 64], 1))
                                #c1 = batch_normalize(c1, is_training)
                            sum = tf.concat([a1,b1,c1],3)
                            #sum = batch_normalize(sum, is_training)
                            with tf.variable_scope('ud4'):
                                x1 = lrelu(deconv_layer(tf.concat([sum,x1],3), [1, 1, 64, 256], [self.batch_size, self.weight, self.height, 64], 1))
                                #x1 = batch_normalize(x1, is_training)
                            with tf.variable_scope('ud5'):
                                x2 = lrelu(deconv_layer(tf.concat([sum,x2],3), [1, 1, 64, 256], [self.batch_size, self.weight, self.height, 64], 1))
                                #x2 = batch_normalize(x2, is_training)
                            with tf.variable_scope('ud6'):
                                x3 = lrelu(deconv_layer(tf.concat([sum,x3],3), [1, 1, 64, 256], [self.batch_size, self.weight, self.height, 64], 1))
                                #x3 = batch_normalize(x3, is_training)
                    with tf.variable_scope('ud7'):
                        block_out = lrelu(deconv_layer(tf.concat([x1, x2, x3],3), [3, 3, 64, 192], [self.batch_size, self.weight, self.height, 64], 1))
                    #x = x1+x2+x3+x
                    x+=block_out
                    #x = batch_normalize(x, is_training))
                    
            #detail = x
            with tf.variable_scope('conv6'):
                x = deconv_layer(
                    x, [3, 3, 256, 64], [self.batch_size, self.weight, self.height, 256], 1)#2
                x = pixel_shuffle_layerg(x, 2, 64) # n_split = 256 / 2 ** 2
                x = lrelu(x)
            
            with tf.variable_scope('conv7'):
                x = deconv_layer(
                    x, [3, 3, 256, 64], [self.batch_size, self.weight*2, self.height*2, 256], 1)#2
                x = pixel_shuffle_layerg(x, 2, 64) # n_split = 256 / 2 ** 2
                x = lrelu(x)
                
            with tf.variable_scope('conv8'):
                x_detail = deconv_layer(
                    x, [3, 3, 3, 64], [self.batch_size, self.weight*4, self.height*4, 3], 1)
            x_srbase = x_detail + self.bic_ref
            
            # frame
            x_fa = self.Laplacian(x_srbase) 
            with tf.variable_scope('conv_e1'):
                x_f = conv_layer(x_fa, [3, 3, 3, 64], 1)
                x_f = lrelu(x_f)
            with tf.variable_scope('conv_e2'):
                x_f = conv_layer(x_f, [3, 3, 64, 64], 1)
                x_f = lrelu(x_f)
            with tf.variable_scope('conv_e3'):
                x_f = conv_layer(x_f, [3, 3, 64, 128], 2)
                x_f = lrelu(x_f)
            with tf.variable_scope('conv_e4'):
                x_f = conv_layer(x_f, [3, 3, 128, 128], 1)
                x_f = lrelu(x_f)
            with tf.variable_scope('conv_e5'):
                x_f = conv_layer(x_f, [3, 3, 128, 256], 2)
                x_f = lrelu(x_f)
            with tf.variable_scope('conv_e7'):
                x_f = conv_layer(x_f, [3, 3, 256, 64], 1)
                x_f = lrelu(x_f)
            with tf.variable_scope('conv_m2'):
                x_f = conv_layer(x_f, [3, 3, 64, 256], 1)
                x_f = lrelu(x_f)
            res_in = x_f
            #res_in = x_f
            # mask
            with tf.variable_scope('conv_m3'):
                x_mask = deconv_layer(
                    res_in, [3, 3, 64, 256], [self.batch_size, self.weight, self.height, 64], 1)
                x_mask = lrelu(x_mask)
            with tf.variable_scope('conv_m4'):# res_in
                x_mask = deconv_layer(
                    x_mask, [3, 3, 128, 64], [self.batch_size, self.weight, self.height, 128], 1)
                x_mask = lrelu(x_mask)
            with tf.variable_scope('conv_m5'):
                x_mask = deconv_layer(
                    x_mask, [3, 3, 256, 128], [self.batch_size, self.weight, self.height, 256], 1)
                x_mask = lrelu(x_mask)

            frame_mask = tf.nn.sigmoid(x_mask)
            x_frame = frame_mask*x_f + x_f
            with tf.variable_scope('conv_m6'):
                x_frame = deconv_layer(
                    x_frame, [3, 3, 64, 256], [self.batch_size, self.weight, self.height, 64], 1)
                x_frame = lrelu(x_frame)
                
            with tf.variable_scope('conv_d3'):
                x_frame = deconv_layer(
                    x_frame, [3, 3, 256, 64], [self.batch_size, self.weight, self.height, 256], 1)
                x_frame = pixel_shuffle_layerg(x_frame, 2, 64)
                x_frame = lrelu(x_frame)
            with tf.variable_scope('conv_d4'):
                x_frame = deconv_layer(
                    x_frame, [3, 3, 256, 64], [self.batch_size, self.weight*2, self.height*2, 256], 1)
                x_frame = pixel_shuffle_layerg(x_frame, 2, 64)
                x_frame = lrelu(x_frame)
             
            #x_de = x_d + x_frame
            with tf.variable_scope('fusion2'):
                x_frame = deconv_layer(
                    x_frame, [3, 3, 3, 64], [self.batch_size, self.weight*4, self.height*4, 3], 1)
                #x_de = lrelu(x_de)

            x_sr = x_frame + x_srbase - x_fa
            frame_e = x_frame-x_fa
        self.g_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        return frame_e, x_srbase, x_sr
        
    def downscale(self, x):
        K = 4
        arr = np.zeros([K, K, 3, 3])
        arr[:, :, 0, 0] = 1.0 / K ** 2
        arr[:, :, 1, 1] = 1.0 / K ** 2
        arr[:, :, 2, 2] = 1.0 / K ** 2
        weight = tf.constant(arr, dtype=tf.float32)
        downscaled = tf.nn.conv2d(
            x, weight, strides=[1, K, K, 1], padding='SAME')
        return downscaled
    
    def sobel(self, x):
        weight=tf.constant([[[-1.0, -1.0, -1.0], [0, 0, 0], [1.0, 1.0, 1.0],
                                  [-2.0, -2.0, -2.0], [0, 0, 0], [2.0, 2.0, 2.0],
                                  [-1.0, -1.0, -1.0], [0, 0, 0], [1.0, 1.0, 1.0]],
                            [[-1.0, -1.0, -1.0], [0, 0, 0], [1.0, 1.0, 1.0],
                                  [-2.0, -2.0, -2.0], [0, 0, 0], [2.0, 2.0, 2.0],
                                  [-1.0, -1.0, -1.0], [0, 0, 0], [1.0, 1.0, 1.0]],
                            [[-1.0, -1.0, -1.0], [0, 0, 0], [1.0, 1.0, 1.0],
                                  [-2.0, -2.0, -2.0], [0, 0, 0], [2.0, 2.0, 2.0],
                                  [-1.0, -1.0, -1.0], [0, 0, 0], [1.0, 1.0, 1.0]]],
                                 shape=[3, 3, 3, 3])
        frame=tf.nn.conv2d(x,weight,[1,1,1,1],padding='SAME')
        frame = tf.cast(((frame - tf.reduce_min(frame)) / (tf.reduce_max(frame) - tf.reduce_min(frame)))*1.0, tf.float32)
        #frame = tf.cast(((frame - tf.reduce_min(frame)) / (tf.reduce_max(frame) - tf.reduce_min(frame))) * 255, tf.uint8)
        return frame
        
    def Laplacian(self, x):
        weight=tf.constant([
        [[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]]],
        [[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],[[8.,0.,0.],[0.,8.,0.],[0.,0.,8.]],[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]]],
        [[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]]]
        ])
        frame=tf.nn.conv2d(x,weight,[1,1,1,1],padding='SAME')
        #frame = tf.cast(((frame - tf.reduce_min(frame)) / (tf.reduce_max(frame) - tf.reduce_min(frame))) * 255, tf.uint8)
        return frame
    
        