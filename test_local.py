# Coypright 2019 IBM. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Dict

# Local imports
import common.util as util
import common.inference_request as inference_request
import handlers

# System imports
import base64
import os
import tensorflow as tf

# Panda pic from Wikimedia; also used in
# https://github.com/tensorflow/models/blob/master/research/slim/nets ...
#   ... /mobilenet/mobilenet_example.ipynb
_PANDA_PIC_URL = ("https://upload.wikimedia.org/wikipedia/commons/f/fe/"
                  "Giant_Panda_in_Beijing_Zoo_1.JPG")

_TMP_DIR = "./temp"

_SAVED_MODEL_DIR = "./saved_model"


def main():
  """
  Spin up a local copy of the model, generate a JSON request, pass that
  through the model, and print the result.
  """
  if not os.path.isdir(_TMP_DIR):
    os.mkdir(_TMP_DIR)

  # Prepare a request
  image_path = util.fetch_or_use_cached(_TMP_DIR, "panda.jpg",
                                        _PANDA_PIC_URL)
  with open(image_path, "rb") as f:
    image_data = f.read()
  thresh = 0.7

  request = inference_request.InferenceRequest()
  request.raw_inputs["image"] = base64.urlsafe_b64encode(
          image_data).decode("utf-8")
  request.raw_inputs["threshold"] = thresh

  # Fire up TensorFlow and perform end-to-end inference
  with tf.Session() as sess:
    graph = tf.Graph()
    with graph.as_default():
      meta_graph = tf.saved_model.loader.load(
        sess,
        [tf.saved_model.tag_constants.SERVING],
        _SAVED_MODEL_DIR)  # type: tf.MetaGraphDef

      # Extract serving "method" signature
      signature = meta_graph.signature_def["serving_default"]

      print("Signature:\n{}".format(signature))

      odh = handlers.ObjectDetectorHandlers()
      odh.pre_process(request)
      inference_request.pass_to_local_tf(request, sess, graph, signature)
      odh.post_process(request)
      print("Result:\n{}".format(request.json_result()))


if __name__ == "__main__":
  main()
