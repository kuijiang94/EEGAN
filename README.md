# Edge Enhanced GAN For Remote Sensing Image Super-Resolution (EEGAN)

This is an implementation of the EEGAN model proposed in the paper
([Edge Enhanced GAN For Remote Sensing Image Super-Resolution]
with TensorFlow.

# Requirements

- Python 3
- TensorFlow 1.1
- OpenCV

# Usage

## I. Pretrain the VGG-19 model

Download the ImageNet dataset and preprocess them with:

```
$ cd vgg19/imagenet
$ python get_urls.py
$ python create_db.py
$ python download_images.py
$ python preprocess.py
```

Train with:

```
$ cd vgg19
$ python train.py
```

Or you can download the pretrained model file:
[vgg19_model.tar.gz](
https://drive.google.com/open?id=0B-s6ok7B0V9vcXNfSzdjZ0lCc0k)


## II. Train the EEGAN model

Download the remote sensing dataset and preprocess them with:

```
$ cd /src/lfw
$ python crop.py
$ python lfw.py
```
to generat the training samples with suitable size.

Train with:

```
$ cd src
$ python train.py
```

The evaluation result will be stored in "src/result".


Test with:

Select the test samples and palce them in test/test30 folder, and then preprocess them with:

```
$ cd src/test
$ python test.py
```
The test result will be stored in "test/result30".



# Loss function

## Adversarial loss 

This implementation adopts the least squares loss function instead  
of the sigmoid cross entropy loss function for the discriminator.
See the details: [Least Squares Generative Adversarial Networks](
https://arxiv.org/abs/1611.04076)

## Content loss

## Consistency loss
