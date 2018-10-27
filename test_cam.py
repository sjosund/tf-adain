#!/usr/bin/env python
from itertools import cycle
import os
from time import sleep

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
from video import WebcamVideoStream


CHANNELS_LAST = 'channels_last'
CHANNELS_FIRST = 'channels_first'


def style_transfer(
    style_path=None,
    style_size=512,
    crop=None,
    preserve_color=None,
    content_size=512,
    alpha=1.0,
    gpu=0,
    vgg_weights='models/vgg19_weights_normalized.h5',
    decoder_weights='models/decoder_weights.h5',
    save_model_to=None,
    load_model_from=None
):
    # Assume that it is either an h5 file or a name of a TensorFlow checkpoint
    decoder_in_h5 = decoder_weights.endswith('.h5')

    if gpu >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        data_format = CHANNELS_FIRST
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        data_format = CHANNELS_LAST

    if load_model_from is not None:
        graph = load_graph(load_model_from)
        content = graph.get_tensor_by_name('prefix/content:0')
        style = graph.get_tensor_by_name('prefix/style:0')
        decoder = graph.get_tensor_by_name('prefix/output:0')
    else:
        graph = tf.Graph()
        content, style, decoder = _build_graph(
            vgg_weights,
            decoder_weights if decoder_in_h5 else None,
            alpha,
            data_format=data_format
        )

    # TODO Why would this be needed????
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('output.avi', fourcc, 15.0, (x_new, y_new))

    cap = WebcamVideoStream(resolution=content_size, src=0)
    # cap = cv2.VideoCapture(0)
    # Set resolution
    # if content_size is not None:
    #     x_length, y_length = content_size, content_size
    #     cap.set(3, x_length)  # 3 and 4 are OpenCV property IDs.
    #     cap.set(4, y_length)
    # x_new = int(cap.get(3))
    # y_new = int(cap.get(4))
    #
    # print('Resolution is: {0} by {1}'.format(x_new, y_new))

    cv2.namedWindow('frame', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)

    # Initial capture of style
    #import ipdb; ipdb.set_trace()
    if style_path is None:
        style_from_cam = True
        style_path = '/tmp/tmp_style_img.jpg'
        while True:
            original = cap.read()
            original = cv2.flip(original, 1)
            cv2.imshow('frame', original)
            if cv2.waitKey(30) & 0xFF == ord('n'):
                original = cap.read()
                print(original.shape)
                cv2.imwrite(style_path, original)
                break
    else:
        style_from_cam = False


    with tf.Session(graph=graph) as sess:
        print("Startnig session")
        style_images = []
        style_images_show = []
        print(style_path)
        if not isinstance(style_path, list):
            style_paths = [style_path]
        else:
            style_paths = style_path
        for style_path in style_paths:
            style_image = load_image(style_path, style_size, crop)
            style_image, style_image_show = read_style_image(
                style_image,
                data_format=data_format)
            style_images_show.append(style_image_show)
            style_images.append(style_image)
        print('length', len(style_images))
        cycler = cycle(style_images)
        cycler_show = cycle(style_images_show)

        current_style_show = next(cycler_show)

        if decoder_in_h5:
            if load_model_from is not None:
                sess.run(tf.global_variables_initializer())
        else:
            saver = tf.train.Saver()
            saver.restore(sess, decoder_weights)

        if save_model_to is not None:
            tf.saved_model.simple_save(
                session=sess,
                export_dir=save_model_to,
                inputs={
                    'style': style,
                    'content': content
                },
                outputs={
                    'output': decoder
                }
            )
            saver = tf.train.Saver()
            dir = '{}model'.format(save_model_to)
            os.makedirs(dir)

            saver.save(sess=sess, save_path='{}/model.ckpt'.format(dir))

            return

        while True:
            original = cap.read()
            content_image = original.astype(np.float32)
            content_image = cv2.cvtColor(content_image, cv2.COLOR_BGR2RGB)

            content_image = prepare_image(content_image, data_format=data_format)
            content_image = content_image[np.newaxis, ...]

            print(content_image.shape)
            img_out = sess.run(
                decoder,
                feed_dict={
                    content: content_image,
                    style: style_image
                }
            )

            img_out = np.clip(img_out * 255, 0, 255)
            img_out = np.squeeze(img_out).astype(np.uint8)
            if data_format == CHANNELS_FIRST:
                img_out = img_out.transpose(1, 2, 0)
            img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
            img_out = cv2.flip(img_out, 1)
            img_out[:100, :100, :] = current_style_show
            cv2.imshow('frame', img_out)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break
            if cv2.waitKey(1) & 0xFF == ord('n'):
                if style_from_cam:
                    cv2.imshow('frame', np.ones_like(img_out))
                    original = cap.read()
                    style_image, current_style_show = read_style_image(
                        original,
                        from_cam=True,
                        data_format=data_format)
                    sleep(1)
                else:
                    style_image = next(cycler)
                    current_style_show = next(cycler_show)
                print(style_image.shape)


def _build_graph(vgg_weights, decoder_weights, alpha, data_format):
    if data_format == CHANNELS_FIRST:
        content = tf.placeholder(shape=(1, 3, None, None), dtype=tf.float32,
                                 name='content')
        style = tf.placeholder(shape=(1, 3, None, None), dtype=tf.float32,
                               name='style')
    else:
        content = tf.placeholder(shape=(1, None, None, 3), dtype=tf.float32,
                                 name='content')
        style = tf.placeholder(shape=(1, None, None, 3), dtype=tf.float32,
                               name='style')

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

    decoder = tf.identity(decoder, name='output')

    return content, style, decoder


def read_style_image(style_image, from_cam=False, data_format=CHANNELS_FIRST):
    if from_cam:
        style_image = cv2.cvtColor(style_image, cv2.COLOR_BGR2RGB)
    style_image = prepare_image(style_image, data_format=data_format)
    style_image_show = cv2.cvtColor(
        imresize(
            np.transpose(style_image, (1, 2, 0)),
            size=(100, 100)
        ),
        cv2.COLOR_BGR2RGB
    )
    style_image = style_image[np.newaxis, ...]
    return style_image, style_image_show


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph


if __name__ == '__main__':
    Fire(style_transfer)
