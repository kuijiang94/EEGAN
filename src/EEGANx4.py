import tensorflow as tf
import numpy as np
import sys
sys.path.append('../utils')
sys.path.append('../vgg19')
from layer import *
from vgg19 import VGG19

class SRGAN:
    def __init__(self, x, is_training, batch_size):
        self.batch_size = batch_size
        self.vgg = VGG19(None, None, None)
        self.downscaled = self.downscale(x)
        self.bic_ref = tf.image.resize_images(self.downscaled, [self.image_size*4, self.image_size*4], method=2)
        self.frame_sr, self.base_sr, self.imitation_sr  = self.generator(self.downscaled, is_training, False)
        self.real_output = self.discriminator(x, is_training, False)
        self.fake_output = self.discriminator(self.base_sr, is_training, True)

        self.g_loss, self.d_loss= self.inference_losses(
            x, self.base_sr, self.imitation_sr,  self.real_output, self.fake_output)
    image_size =  24

    def generator(self, x, is_training, reuse):
        with tf.variable_scope('generator', reuse=reuse):
            input = x
            with tf.variable_scope('conv1'):
                x = deconv_layer(
                    input, [3, 3, 128, 3], [self.batch_size, self.image_size, self.image_size, 128], 1)
                x = lrelu(x)
            with tf.variable_scope('conv2'):
                x = deconv_layer(
                    x, [3, 3, 64, 128], [self.batch_size, self.image_size, self.image_size, 64], 1)
                x = lrelu(x)
            #shortcut = x
            for i in range(6):
                with tf.variable_scope('block{}ex1'.format(i+1)):
                    x1=x2=x3=x
                    for j in range(3):
                        with tf.variable_scope('block{}_{}ex1'.format(i+1,j+1)):
                            with tf.variable_scope('ud1'):
                                a1 = lrelu(deconv_layer(x1, [3, 3, 64, 64], [self.batch_size, self.image_size, self.image_size, 64], 1))
                                #a1 = batch_normalize(a1, is_training)
                            with tf.variable_scope('ud2'):
                                b1 = lrelu(deconv_layer(x2, [3, 3, 64, 64], [self.batch_size, self.image_size, self.image_size, 64], 1))
                                #b1 = batch_normalize(b1, is_training)
                            with tf.variable_scope('ud3'):
                                c1 = lrelu(deconv_layer(x3, [3, 3, 64, 64], [self.batch_size, self.image_size, self.image_size, 64], 1))
                                #c1 = batch_normalize(c1, is_training)
                            sum = tf.concat([a1,b1,c1],3)
                            #sum = batch_normalize(sum, is_training)
                            with tf.variable_scope('ud4'):
                                x1 = lrelu(deconv_layer(tf.concat([sum,x1],3), [1, 1, 64, 256], [self.batch_size, self.image_size, self.image_size, 64], 1))
                                #x1 = batch_normalize(x1, is_training)
                            with tf.variable_scope('ud5'):
                                x2 = lrelu(deconv_layer(tf.concat([sum,x2],3), [1, 1, 64, 256], [self.batch_size, self.image_size, self.image_size, 64], 1))
                                #x2 = batch_normalize(x2, is_training)
                            with tf.variable_scope('ud6'):
                                x3 = lrelu(deconv_layer(tf.concat([sum,x3],3), [1, 1, 64, 256], [self.batch_size, self.image_size, self.image_size, 64], 1))
                                #x3 = batch_normalize(x3, is_training)
                    with tf.variable_scope('ud7'):
                        block_out = lrelu(deconv_layer(tf.concat([x1, x2, x3],3), [3, 3, 64, 192], [self.batch_size, self.image_size, self.image_size, 64], 1))
                    #x = x1+x2+x3+x
                    x+=block_out
                    #x = batch_normalize(x, is_training))
                    
            #detail = x
            with tf.variable_scope('conv6'):
                x = deconv_layer(
                    x, [3, 3, 256, 64], [self.batch_size, self.image_size, self.image_size, 256], 1)#2
                x = pixel_shuffle_layerg(x, 2, 64) # n_split = 256 / 2 ** 2
                x = lrelu(x)
            
            with tf.variable_scope('conv7'):
                x = deconv_layer(
                    x, [3, 3, 256, 64], [self.batch_size, self.image_size*2, self.image_size*2, 256], 1)#2
                x = pixel_shuffle_layerg(x, 2, 64) # n_split = 256 / 2 ** 2
                x = lrelu(x)
                
            with tf.variable_scope('conv8'):
                x_detail = deconv_layer(
                    x, [3, 3, 3, 64], [self.batch_size, self.image_size*4, self.image_size*4, 3], 1)
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
            res_in = x_f
            # frame
            for i in range(3):
                with tf.variable_scope('block{}ex2'.format(i+1)):
                    x1=x2=x3=x_f
                    for j in range(3):
                        with tf.variable_scope('block{}_{}ex1'.format(i+1,j+1)):
                            with tf.variable_scope('ud1'):
                                a1 = lrelu(deconv_layer(x1, [3, 3, 64, 64], [self.batch_size, self.image_size, self.image_size, 64], 1))
                                #a1 = batch_normalize(a1, is_training)
                            with tf.variable_scope('ud2'):
                                b1 = lrelu(deconv_layer(x2, [3, 3, 64, 64], [self.batch_size, self.image_size, self.image_size, 64], 1))
                                #b1 = batch_normalize(b1, is_training)
                            with tf.variable_scope('ud3'):
                                c1 = lrelu(deconv_layer(x3, [3, 3, 64, 64], [self.batch_size, self.image_size, self.image_size, 64], 1))
                                #c1 = batch_normalize(c1, is_training)
                            sum = tf.concat([a1,b1,c1],3)
                            #sum = batch_normalize(sum, is_training)
                            with tf.variable_scope('ud4'):
                                x1 = lrelu(deconv_layer(tf.concat([sum,x1],3), [1, 1, 64, 256], [self.batch_size, self.image_size, self.image_size, 64], 1))
                                #x1 = batch_normalize(x1, is_training)
                            with tf.variable_scope('ud5'):
                                x2 = lrelu(deconv_layer(tf.concat([sum,x2],3), [1, 1, 64, 256], [self.batch_size, self.image_size, self.image_size, 64], 1))
                                #x2 = batch_normalize(x2, is_training)
                            with tf.variable_scope('ud6'):
                                x3 = lrelu(deconv_layer(tf.concat([sum,x3],3), [1, 1, 64, 256], [self.batch_size, self.image_size, self.image_size, 64], 1))
                                #x3 = batch_normalize(x3, is_training)
                    with tf.variable_scope('ud7'):
                        block_out = lrelu(deconv_layer(tf.concat([x1, x2, x3],3), [3, 3, 64, 192], [self.batch_size, self.image_size, self.image_size, 64], 1))
                    #x = x1+x2+x3+x
                    x_f+=block_out
            with tf.variable_scope('conv_e8'):
                x_f = conv_layer(x_f, [3, 3, 64, 256], 1)
                x_f = lrelu(x_f)
            #res_in = x_f
            # mask
            with tf.variable_scope('conv_m3'):
                x_mask = deconv_layer(
                    res_in, [3, 3, 64, 64], [self.batch_size, self.image_size, self.image_size, 64], 1)
                x_mask = lrelu(x_mask)
            with tf.variable_scope('conv_m4'):# res_in
                x_mask = deconv_layer(
                    x_mask, [3, 3, 128, 64], [self.batch_size, self.image_size, self.image_size, 128], 1)
                x_mask = lrelu(x_mask)
            with tf.variable_scope('conv_m5'):
                x_mask = deconv_layer(
                    x_mask, [3, 3, 256, 128], [self.batch_size, self.image_size, self.image_size, 256], 1)
                x_mask = lrelu(x_mask)

            frame_mask = tf.nn.sigmoid(x_mask)
            x_frame = frame_mask*x_f + x_f
            with tf.variable_scope('conv_m6'):
                x_frame = deconv_layer(
                    x_frame, [3, 3, 64, 256], [self.batch_size, self.image_size, self.image_size, 64], 1)
                x_frame = lrelu(x_frame)
                
            with tf.variable_scope('conv_d3'):
                x_frame = deconv_layer(
                    x_frame, [3, 3, 256, 64], [self.batch_size, self.image_size, self.image_size, 256], 1)
                x_frame = pixel_shuffle_layerg(x_frame, 2, 64)
                x_frame = lrelu(x_frame)
            with tf.variable_scope('conv_d4'):
                x_frame = deconv_layer(
                    x_frame, [3, 3, 256, 64], [self.batch_size, self.image_size*2, self.image_size*2, 256], 1)
                x_frame = pixel_shuffle_layerg(x_frame, 2, 64)
                x_frame = lrelu(x_frame)
             
            #x_de = x_d + x_frame
            with tf.variable_scope('fusion2'):
                x_frame = deconv_layer(
                    x_frame, [3, 3, 3, 64], [self.batch_size, self.image_size*4, self.image_size*4, 3], 1)
                #x_de = lrelu(x_de)

            x_sr = x_frame + x_srbase - x_fa
            frame_e = x_frame-x_fa
        self.g_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        return frame_e, x_srbase, x_sr
        
    def discriminator(self, x, is_training, reuse):
        with tf.variable_scope('discriminator', reuse=reuse):
            with tf.variable_scope('conv1'):
                x = conv_layer(x, [3, 3, 3, 64], 1)
                x = lrelu(x)
                #x = batch_normalize(x, is_training)
            with tf.variable_scope('conv2'):
                x = conv_layer(x, [3, 3, 64, 64], 1)
                x = lrelu(x)
                #x = batch_normalize(x, is_training)
            #x = avg_pooling_layer(x, 2, 2)
            with tf.variable_scope('conv3'):
                x = conv_layer(x, [3, 3, 64, 128], 1)
                x = lrelu(x)
                #x = batch_normalize(x, is_training)
            #x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            x = avg_pooling_layer(x, 2, 2)
            with tf.variable_scope('conv5'):
                x = conv_layer(x, [3, 3, 128, 256], 1)
                x = lrelu(x)
                #x = batch_normalize(x, is_training)
            x = avg_pooling_layer(x, 2, 2)
            with tf.variable_scope('conv6'):
                x = conv_layer(x, [3, 3, 256, 512], 1)
                x = lrelu(x)
                #x = batch_normalize(x, is_training)
            x = avg_pooling_layer(x, 2, 2)
            with tf.variable_scope('conv7'):
                x = conv_layer(x, [3, 3, 512, 1024], 1)
                x = lrelu(x)
                #x = batch_normalize(x, is_training)
            x = tf.reshape(x, (-1, 1, 1, (self.image_size//2)*(self.image_size//2)*1024))
            x = flatten_layer(x)
            with tf.variable_scope('fc1'):
                x = full_connection_layer(x, 1024)
                x = lrelu(x)
            with tf.variable_scope('softmax'):
                x = full_connection_layer(x, 1)
                
        self.d_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        return x
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
        frame = tf.cast(((frame - tf.reduce_min(frame)) / (tf.reduce_max(frame) - tf.reduce_min(frame)))*2.0-1, tf.float32)
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
    
        
    def inference_losses(self, x, base_sr, imitation_sr, true_output, fake_output):
        def inference_content_loss(x, imitation):#frame_hr, frame_sr,, real_frame, fake_frame,real_base, fake_base,
            _, x_phi = self.vgg.build_model(
                x, tf.constant(False), False) # First
            _, imitation_phi = self.vgg.build_model(
                imitation, tf.constant(False), True) # Second

            content_loss = None
            for i in range(len(x_phi)):
                #loss = tf.nn.l2_loss(x_phi[i] - imitation_phi[i])
                loss = tf.reduce_mean(tf.sqrt((x_phi[i] - imitation_phi[i]) ** 2+(1e-3)**2))
                if content_loss is None:
                    content_loss = loss
                else:
                    content_loss = content_loss + loss
            return tf.reduce_mean(content_loss)
        
        def inference_content_loss_sr(frame_hr, frame_sr):
            content_base_loss = tf.reduce_mean(tf.sqrt((frame_hr - frame_sr) ** 2+(1e-3)**2))
            return tf.reduce_mean(content_base_loss)

        def inference_adversarial_loss(true_output, fake_output):
            alpha = 1e-5#1e-2,1e-5
            g_loss = tf.reduce_mean(tf.sqrt((fake_output - tf.ones_like(fake_output)) ** 2+(1e-3)**2))
            d_loss_real = tf.reduce_mean(tf.sqrt((true_output - tf.ones_like(true_output)) ** 2+(1e-3)**2))
            d_loss_fake = tf.reduce_mean(tf.sqrt((fake_output + tf.ones_like(fake_output)) ** 2+(1e-3)**2))
            d_loss = d_loss_real + d_loss_fake
            return (g_loss * alpha, d_loss * alpha)

        def inference_adversarial_loss_with_sigmoid(real_frame, fake_frame):
            alpha = 1e-3#1e-3
            g_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(fake_frame),
                logits=fake_frame)
            d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(real_frame),
                logits=real_frame)
            d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(fake_frame),
                logits=fake_frame)
            d_loss = d_loss_real + d_loss_fake
            return (g_loss * alpha, d_loss * alpha)

        content_loss = inference_content_loss(x, base_sr)
        content_sr_loss = inference_content_loss_sr(x, imitation_sr)
        generator_loss, discriminator_loss = (inference_adversarial_loss(true_output, fake_output))
        g_loss = content_loss + generator_loss + 10*content_sr_loss
        d_loss = discriminator_loss
        return (g_loss, d_loss)

