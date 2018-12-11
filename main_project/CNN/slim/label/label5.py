import numpy as np
import os
import tensorflow as tf
import subprocess

from nets import inception
from preprocessing import inception_preprocessing

checkpoints_dir = '/home/vb/Desktop/inception_resnet_v2/img_from_server/ckpt'

tf.app.flags.DEFINE_string(
    'tmp_image_path', '/home/vb/Desktop/inception_resnet_v2/img_from_server/tmp.jpg',
    'Path Error : check image file path')

tf.app.flags.DEFINE_string(
    'org_image_path', 'None',
    'Path Error : check image file path')

FLAGS = tf.app.flags.FLAGS

slim1 = tf.contrib.slim
slim2 = tf.contrib.slim
slim3 = tf.contrib.slim

image_size = inception.inception_resnet_v2.default_image_size    
tmp_img_path = FLAGS.tmp_image_path
org_img_path = FLAGS.org_image_path
f = open('/home/vb/Desktop/inception_resnet_v2/img_from_server/tag_info.txt', "w")

def get_img_inf(ckpt_dir, ckpt_name, probabilities):
    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(ckpt_dir,ckpt_name),
        slim.get_model_variables('InceptionResnetV2'))
    
    with tf.Session() as sess:
        init_fn(sess)
        np_image, probabilities = sess.run([image, probabilities])
        probabilities = probabilities[0, 0:]
        sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x:x[1])]
    
    return sorted_inds,probabilities


cpy_cmd = 'cp %s %s' % (org_img_path, tmp_img_path)
os.system(cpy_cmd)

sorted_inds_list = []
names_list = []
prob_list = []

names_list.append(os.listdir("/home/vb/Desktop/inception_resnet_v2/dataset/data_tag1/image/"))
with tf.Graph().as_default():

    image_input = tf.read_file(org_img_path)
    image = tf.image.decode_jpeg(image_input, channels=3)
    processed_image = inception_preprocessing.preprocess_image(image,
                                                         image_size,
                                                         image_size,
                                                         is_training=False)
    processed_images  = tf.expand_dims(processed_image, 0)
    
    with slim1.arg_scope(inception.inception_resnet_v2_arg_scope()):
        logits, _ = inception.inception_resnet_v2(processed_images, num_classes=10, is_training=False)
    probabilities = tf.nn.softmax(logits)

    init_fn = slim1.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir,'model.ckpt-1'),
        slim1.get_model_variables('InceptionResnetV2'))
    
    with tf.Session() as sess:
        init_fn(sess)
        np_image, probabilities = sess.run([image, probabilities])
        prob_list.append(probabilities[0, 0:])
        sorted_inds_list.append([i[0] for i in sorted(enumerate(-prob_list[0]), key=lambda x:x[1])])


names_list.append(os.listdir("/home/vb/Desktop/inception_resnet_v2/dataset/data_tag3/image/"))
with tf.Graph().as_default():

    image_input = tf.read_file(org_img_path)
    image = tf.image.decode_jpeg(image_input, channels=3)
    processed_image = inception_preprocessing.preprocess_image(image,
                                                         image_size,
                                                         image_size,
                                                         is_training=False)
    processed_images  = tf.expand_dims(processed_image, 0)
    
    with slim2.arg_scope(inception.inception_resnet_v2_arg_scope()):
        logits, _ = inception.inception_resnet_v2(processed_images, num_classes=7, is_training=False)
    probabilities = tf.nn.softmax(logits)

    init_fn = slim2.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir,'model.ckpt-2'),
        slim2.get_model_variables('InceptionResnetV2'))
    
    with tf.Session() as sess:
        init_fn(sess)
        np_image, probabilities = sess.run([image, probabilities])
        prob_list.append(probabilities[0, 0:])
        sorted_inds_list.append([i[0] for i in sorted(enumerate(-prob_list[1]), key=lambda x:x[1])])

names_list.append(os.listdir("/home/vb/Desktop/inception_resnet_v2/dataset/data_tag2/image/"))
with tf.Graph().as_default():

    image_input = tf.read_file(org_img_path)
    image = tf.image.decode_jpeg(image_input, channels=3)
    processed_image = inception_preprocessing.preprocess_image(image,
                                                         image_size,
                                                         image_size,
                                                         is_training=False)
    processed_images  = tf.expand_dims(processed_image, 0)
    
    with slim3.arg_scope(inception.inception_resnet_v2_arg_scope()):
        logits, _ = inception.inception_resnet_v2(processed_images, num_classes=4, is_training=False)
    probabilities = tf.nn.softmax(logits)

    init_fn = slim3.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir,'model.ckpt-3'),
        slim3.get_model_variables('InceptionResnetV2'))

    with tf.Session() as sess:
        init_fn(sess)
        np_image, probabilities = sess.run([image, probabilities])
        prob_list.append(probabilities[0, 0:])
        sorted_inds_list.append([i[0] for i in sorted(enumerate(-prob_list[2]), key=lambda x:x[1])])


ret_str = '[' 
for n in range(len(names_list)):
    names_list[n].sort()
	
    ret_str += '"tag_field%d":[' % n
    for i in range(5):
        index = sorted_inds_list[n][i]
        ret_str += '{"tag":"%s","Probability":"%0.2f"}' % (names_list[n][index], prob_list[n][index])
        if i != 4:
            ret_str+=','
    ret_str += ']'
    if n != len(names_list)-1:
        ret_str+=','
ret_str += ']'
f.write(ret_str)
print(ret_str)
os.system('rm %s' % tmp_img_path)
