# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Downloads and converts a particular dataset.

$ python download_and_convert_data.py \
    --dataset_name=nsdevil \
    --dataset_dir=/tmp/nsdevil
```

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from datasets import convert_nsdevil1
from datasets import convert_nsdevil2
from datasets import convert_nsdevil3

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'dataset_name',
   'nsdevil1',
    'The name of the dataset to convert, "nsdevil".')

tf.app.flags.DEFINE_string(
    'dataset_dir',
    None,
    'The directory where the output TFRecords and temporary files are saved.')


def main(_):
  if not FLAGS.dataset_name:
    raise ValueError('You must supply the dataset name with --dataset_name')
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')
  if FLAGS.dataset_name=='nsdevil1':
    convert_nsdevil1.run(FLAGS.dataset_dir)
  if FLAGS.dataset_name=='nsdevil2':
    convert_nsdevil2.run(FLAGS.dataset_dir)
  if FLAGS.dataset_name=='nsdevil3':
    convert_nsdevil3.run(FLAGS.dataset_dir)

if __name__ == '__main__':
  tf.app.run()
