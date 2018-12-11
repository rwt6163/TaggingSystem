import numpy as np
import os
import tensorflow as tf
import subprocess

from nets import inception
from preprocessing import inception_preprocessing

checkpoints_dir = '/home/vb/Desktop/inception_resnet_v2/img_from_server/ckpt'

tf.app.flags.DEFINE_string(
    'tmp_image_path', 'None',
    'Path Error : check image file path')

tf.app.flags.DEFINE_string(
    'org_image_path', '/home/vb/Desktop/inception_resnet_v2/test.jpg',
    'Path Error : check image file path')

FLAGS = tf.app.flags.FLAGS

slim = tf.contrib.slim
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

with tf.Graph().as_default():

    image_input = tf.read_file(org_img_path)
    image = tf.image.decode_jpeg(image_input, channels=3)
    processed_image = inception_preprocessing.preprocess_image(image,
                                                         image_size,
                                                         image_size,
                                                         is_training=False)
    processed_images  = tf.expand_dims(processed_image, 0)
    prob = []
    with slim.arg_scope(inception.inception_resnet_v2_arg_scope()):
        logits1, _ = inception.inception_resnet_v2(processed_images, num_classes=10, is_training=False)
    prob.append(tf.nn.softmax(logits1))

    with slim.arg_scope(inception.inception_resnet_v2_arg_scope()):
        logits2, _ = inception.inception_resnet_v2(processed_images, num_classes=10, is_training=False)
    prob.append(tf.nn.softmax(logits2))

    with slim.arg_scope(inception.inception_resnet_v2_arg_scope()):
        logits3, _ = inception.inception_resnet_v2(processed_images, num_classes=10, is_training=False, reuse=tf.AUTO_REUSE)
    prob.append(tf.nn.softmax(logits3))

    init_fn = []
    init_fn.append(slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir,'model.ckpt-1'),
        slim.get_model_variables('InceptionResnetV2')))
    
    init_fn.append(slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir,'model.ckpt-2'),
        slim.get_model_variables('InceptionResnetV2')))

    init_fn.append(slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir,'model.ckpt-3'),
        slim.get_model_variables('InceptionResnetV2')))
    
    sorted_inds_list = []
    names_list = []
    prob_list = []
    with tf.Session() as sess:
        for n in range(3):
            init_fn[n](sess)
            np_image, probabilities = sess.run([image, prob[n]])
            prob_list.append(probabilities[0, 0:])
            sorted_inds_list.append([i[0] for i in sorted(enumerate(-prob_list[n]), key=lambda x:x[1])])

    names_list.append(os.listdir("/home/vb/Desktop/inception_resnet_v2/dataset/data_tag1/image/"))
    names_list.append(os.listdir("/home/vb/Desktop/inception_resnet_v2/dataset/data_tag3/image/"))
    names_list.append(os.listdir("/home/vb/Desktop/inception_resnet_v2/dataset/data_tag2/image/"))
    ret_str = '[' 
    for n in range(3):
        names_list[n].sort()
 	
        ret_str += '"tag_field%d":[' % n
        for i in range(5):
            index = sorted_inds_list[n][i]
            ret_str += '{"tag":"%s","Probability":"%0.2f"}' % (names_list[n][index], prob_list[n][index])
            if i != 4:
                ret_str+=','
        ret_str += ']'
        if n != 2:
            ret_str+=','
ret_str += ']'
f.write(ret_str)
print(ret_str)
os.system('rm %s' % tmp_img_path)
