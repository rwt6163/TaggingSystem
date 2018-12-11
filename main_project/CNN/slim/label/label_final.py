from flask import Flask, request, Response
import os
import numpy as np
import cv2

import numpy as np
import os
import tensorflow as tf
import subprocess

from nets import inception
from preprocessing import inception_preprocessing

app = Flask(__name__)


@app.route('/api/test', methods=['POST'])
def hello_world():
    r = request
    nparr = np.fromstring(r.data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_path = "/home/vb/Desktop/inception_resnet_v2/img_from_server/tmp.jpg"
    cv2.imwrite(img_path, img)
    print("누가 접속 했냐 이 씌발럼아")
    
    get_inf(img_path)
    f = open('/home/vb/Desktop/inception_resnet_v2/img_from_server/tag_info.txt', "w")
    ret_str = str(f.read())
    print(ret_str)
    return "rtrt"


def get_inf(img_path):
    checkpoints_dir = '/home/vb/Desktop/inception_resnet_v2/img_from_server/ckpt'
    
    FLAGS = tf.app.flags.FLAGS
    
    slim1 = tf.contrib.slim
    slim2 = tf.contrib.slim
    slim3 = tf.contrib.slim
    
    image_size = inception.inception_resnet_v2.default_image_size
    f = open('/home/vb/Desktop/inception_resnet_v2/img_from_server/tag_info.txt', "w")

    org_img_path = img_path
    
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
        processed_images = tf.expand_dims(processed_image, 0)
    
        with slim1.arg_scope(inception.inception_resnet_v2_arg_scope()):
            logits, _ = inception.inception_resnet_v2(processed_images, num_classes=10, is_training=False)
        probabilities = tf.nn.softmax(logits)
    
        init_fn = slim1.assign_from_checkpoint_fn(
            os.path.join(checkpoints_dir, 'model.ckpt-1'),
            slim1.get_model_variables('InceptionResnetV2'))
    
        with tf.Session() as sess:
            init_fn(sess)
            np_image, probabilities = sess.run([image, probabilities])
            prob_list.append(probabilities[0, 0:])
            sorted_inds_list.append([i[0] for i in sorted(enumerate(-prob_list[0]), key=lambda x: x[1])])
    
    names_list.append(os.listdir("/home/vb/Desktop/inception_resnet_v2/dataset/data_tag3/image/"))
    with tf.Graph().as_default():
        image_input = tf.read_file(org_img_path)
        image = tf.image.decode_jpeg(image_input, channels=3)
        processed_image = inception_preprocessing.preprocess_image(image,
                                                                   image_size,
                                                                   image_size,
                                                                   is_training=False)
        processed_images = tf.expand_dims(processed_image, 0)
    
        with slim2.arg_scope(inception.inception_resnet_v2_arg_scope()):
            logits, _ = inception.inception_resnet_v2(processed_images, num_classes=7, is_training=False)
        probabilities = tf.nn.softmax(logits)
    
        init_fn = slim2.assign_from_checkpoint_fn(
            os.path.join(checkpoints_dir, 'model.ckpt-2'),
            slim2.get_model_variables('InceptionResnetV2'))
    
        with tf.Session() as sess:
            init_fn(sess)
            np_image, probabilities = sess.run([image, probabilities])
            prob_list.append(probabilities[0, 0:])
            sorted_inds_list.append([i[0] for i in sorted(enumerate(-prob_list[1]), key=lambda x: x[1])])
    
    names_list.append(os.listdir("/home/vb/Desktop/inception_resnet_v2/dataset/data_tag2/image/"))
    with tf.Graph().as_default():
        image_input = tf.read_file(org_img_path)
        image = tf.image.decode_jpeg(image_input, channels=3)
        processed_image = inception_preprocessing.preprocess_image(image,
                                                                   image_size,
                                                                   image_size,
                                                                   is_training=False)
        processed_images = tf.expand_dims(processed_image, 0)
    
        with slim3.arg_scope(inception.inception_resnet_v2_arg_scope()):
            logits, _ = inception.inception_resnet_v2(processed_images, num_classes=4, is_training=False)
        probabilities = tf.nn.softmax(logits)
    
        init_fn = slim3.assign_from_checkpoint_fn(
            os.path.join(checkpoints_dir, 'model.ckpt-3'),
            slim3.get_model_variables('InceptionResnetV2'))
    
        with tf.Session() as sess:
            init_fn(sess)
            np_image, probabilities = sess.run([image, probabilities])
            prob_list.append(probabilities[0, 0:])
            sorted_inds_list.append([i[0] for i in sorted(enumerate(-prob_list[2]), key=lambda x: x[1])])
    
    ret_str = ''
    for n in range(len(names_list)):
        names_list[n].sort()
        index = sorted_inds_list[n][0]
        ret_str += '#%s' % names_list[n][index]
    
    f.write(ret_str)

if __name__ == '__main__':
    app.run("0.0.0.0", 8080)
