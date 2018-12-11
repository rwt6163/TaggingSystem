import numpy as np
import os
import tensorflow as tf
import subprocess

from nets import inception
from preprocessing import inception_preprocessing

checkpoints_dir = '/home/vb/Desktop/inception_resnet_v2/train_tag1/'

tf.app.flags.DEFINE_string(
    'tmp_image_path', '/home/vb/Desktop/inception_resnet_v2/img_from_server/tmp.jpg',
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
    
    with slim.arg_scope(inception.inception_resnet_v2_arg_scope()):
        logits, _ = inception.inception_resnet_v2(processed_images, num_classes=10, is_training=False)
    probabilities = tf.nn.softmax(logits)

    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir,'model.ckpt-40000'),
        slim.get_model_variables('InceptionResnetV2'))
    
    with tf.Session() as sess:
        init_fn(sess)
        np_image, probabilities = sess.run([image, probabilities])
        probabilities = probabilities[0, 0:]
        sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x:x[1])]

    names = os.listdir("/home/vb/Desktop/inception_resnet_v2/dataset/old_dataset/image/")
    names.sort()
 	
    ret_str = '['
    for i in range(5):
        index = sorted_inds[i]
        ret_str += '{"tag":"%s","Probability":"%0.2f"}' % (names[index], probabilities[index])
        if i != 4:
            ret_str+=','
ret_str += ']'
f.write(ret_str)
print(ret_str)
os.system('rm %s' % tmp_img_path)
