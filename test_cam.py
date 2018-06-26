#!/usr/bin/env python
from itertools import cycle
import os

import cv2
from fire import Fire
import numpy as np
from scipy.misc import imresize
import tensorflow as tf

from adain.image import load_image, prepare_image
from adain.coral import coral
from adain.nn import build_vgg, build_decoder
from adain.norm import adain
from adain.weights import open_weights


def style_transfer(
        style_path,
        style_size=512,
        crop=None,
        preserve_color=None,
        content_size=512,
        alpha=1.0,
        gpu=0,
        vgg_weights='models/vgg19_weights_normalized.h5',
        decoder_weights='models/decoder_weights.h5'):
    # Assume that it is either an h5 file or a name of a TensorFlow checkpoint
    decoder_in_h5 = decoder_weights.endswith('.h5')
    print(type(style_path))

    if gpu >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        data_format = 'channels_first'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        data_format = 'channels_last'

    content, style, decoder = _build_graph(
        vgg_weights,
        decoder_weights if decoder_in_h5 else None,
        alpha,
        data_format=data_format
    )

    # TODO Why would this be needed????
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('output.avi', fourcc, 15.0, (x_new, y_new))

    cap = cv2.VideoCapture(0)
    # Set resolution
    if content_size is not None:
        x_length, y_length = content_size, content_size
        cap.set(3, x_length)  # 3 and 4 are OpenCV property IDs.
        cap.set(4, y_length)
    x_new = int(cap.get(3))
    y_new = int(cap.get(4))

    print('Resolution is: {0} by {1}'.format(x_new, y_new))

    cv2.namedWindow('frame', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)

    with tf.Session() as sess:
        style_images = []
        style_images_show = []
        print(style_path)
        if not isinstance(style_path, list):
            style_paths = [style_path]
        else:
            style_paths = style_path
        for style_path in style_paths:
            style_image = load_image(style_path, style_size, crop)
            style_image = prepare_image(style_image)
            style_images_show.append(cv2.cvtColor(
                imresize(
                    np.transpose(style_image, (1, 2, 0)),
                    size=(100, 100)
                ),
                cv2.COLOR_BGR2RGB
            ))
            style_image = style_image[np.newaxis, ...]
            style_images.append(style_image)
        print('lenght', len(style_images))
        cycler = cycle(style_images)
        cycler_show = cycle(style_images_show)

        current_style = next(cycler)
        current_style_show = next(cycler_show)

        if decoder_in_h5:
            sess.run(tf.global_variables_initializer())
        else:
            saver = tf.train.Saver()
            saver.restore(sess, decoder_weights)

        while True:
            # TODO Load from webcam
            # TODO If style ticker changed, reload style image.

            ret, original = cap.read()
            content_image = original.astype(np.float32)

            # TODO Maybe put back later
            # if preserve_color:
            #     style_image = coral(style_image, content_image)
            content_image = prepare_image(content_image)
            content_image = content_image[np.newaxis, ...]

            # TODO Bundle together to one sess.run
            img_out = sess.run(
                decoder,
                feed_dict={
                    content: content_image,
                    style: style_image
                }
            )

            img_out = np.clip(img_out * 255, 0, 255)
            img_out = np.squeeze(img_out).astype(np.uint8)
            img_out = cv2.cvtColor(img_out.transpose(1, 2, 0), cv2.COLOR_BGR2RGB)
            img_out = cv2.flip(img_out, 1)
            img_out[:100, :100, :] = current_style_show
            cv2.imshow('frame', img_out)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break
            if cv2.waitKey(1) & 0xFF == ord('n'):
                style_image = next(cycler)
                current_style_show = next(cycler_show)
                print(style_image.shape)


def _build_graph(vgg_weights, decoder_weights, alpha, data_format):
    if data_format == 'channels_first':
        content = tf.placeholder(shape=(1, 3, None, None), dtype=tf.float32)
        style = tf.placeholder(shape=(1, 3, None, None), dtype=tf.float32)
    else:
        content = tf.placeholder(shape=(1, None, None, 3), dtype=tf.float32)
        style = tf.placeholder(shape=(1, None, None, 3), dtype=tf.float32)

    with open_weights(vgg_weights) as w:
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            vgg_content = build_vgg(content, w, data_format=data_format)
            vgg_style = build_vgg(style, w, data_format=data_format)
            content_feature = vgg_content['conv4_1']
            style_feature = vgg_style['conv4_1']

    target = adain(content_feature, style_feature, data_format=data_format)
    weighted_target = target * alpha + (1 - alpha) * content_feature

    if decoder_weights:
        with open_weights(decoder_weights) as w:
            decoder = build_decoder(weighted_target, w, trainable=False,
                                    data_format=data_format)
    else:
        decoder = build_decoder(weighted_target, None, trainable=False,
                                data_format=data_format)

    return content, style, decoder


if __name__ == '__main__':
    Fire(style_transfer)
