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

from common.prepost import PrePost, GraphGen
from common.inference_request import InferenceRequest
from common import util

import re
import tarfile
import tensorflow as tf


################################################################################
# CONSTANTS
_CACHE_DIR = "./cached_files"
_LONG_MODEL_NAME = "ssd_mobilenet_v1_coco_2018_01_28"
_MODEL_TARBALL_URL = ("http://download.tensorflow.org/models/object_detection/"
                      + _LONG_MODEL_NAME + ".tar.gz")

# Label map for decoding label IDs in the output of the graph
_LABEL_MAP_URL = ("https://raw.githubusercontent.com/tensorflow/models/"
                  "f87a58cd96d45de73c9a8330a06b2ab56749a7fa/research/"
                  "object_detection/data/mscoco_label_map.pbtxt")
_FROZEN_GRAPH_MEMBER = _LONG_MODEL_NAME + "/frozen_inference_graph.pb"

################################################################################
# CALLBACKS THAT CREATE GRAPHS
class GraphGenerators(GraphGen):

  def frozen_graph(self):
    # type: () -> tf.GraphDef
    """
    Generates and returns the core TensorFlow graph for the model as a frozen
    (i.e. all variables converted to constants) GraphDef protocol buffer
    message.
    """
    tarball = util.fetch_or_use_cached(_CACHE_DIR,
                                       "{}.tar.gz".format(_LONG_MODEL_NAME),
                                       _MODEL_TARBALL_URL)

    print("Original model files at {}".format(tarball))
    with tarfile.open(tarball) as t:
      frozen_graph_bytes = t.extractfile(_FROZEN_GRAPH_MEMBER).read()
      return tf.GraphDef.FromString(frozen_graph_bytes)

  def input_node_names(self):
    # type: () -> List[str]
    """
    Returns a list of the names of Placeholder ops (AKA nodes) in the graph
    returned by `frozen_graph` that are required inputs for inference.
    """
    return ["image_tensor"]

  def output_node_names(self):
    """
    Returns a list of the names of  ops (AKA nodes) in the graph returned by
    `frozen_graph` that produce output values for inference requests.
    """
    return ["detection_boxes", "detection_classes",
            "detection_scores", "num_detections"]

  def pre_processing_graph(self):
    # type: () -> tf.Graph
    """
    Generates and returns a TensorFlow graph containing preprocessing
    operations. By convention, this graph contains one or more input
    placeholders that correspond to input placeholders by the same name in
    the main graph.

    For each placeholder in the original graph that needs preprocessing,
    the preprocessing graph should contain a placeholder with the same name
    and a second op named "<name of placeholder>_preprocessed", where `<name
    of placeholder>` is the name of the Placeholder op.
    """
    # Preprocessing steps performed:
    # 1. Decode base64
    # 2. Uncompress JPEG/PNG/GIF image file
    # 3. Massage into a single-image batch
    # 2 and 3 are handled by the same op
    img_decode_g = tf.Graph()
    with img_decode_g.as_default():
      raw_image = tf.placeholder(tf.string, name="image_tensor")

      binary_image = tf.io.decode_base64(raw_image)

      # tf.image.decode_image() returns a 4D tensor when it receives a GIF and
      # a 3D tensor for every other file type. This means that you need
      # complicated shape-checking and reshaping logic downstream
      # for it to be of any use in an inference context.
      # So we use decode_gif, which in spite of its name, also handles JPEG and
      # PNG files; and which always returns a batch of images.
      decoded_image_batch = tf.image.decode_gif(
        binary_image, name="image_tensor_preprocessed")
    return img_decode_g

  def post_processing_graph(self):
    # type: () -> tf.Graph
    """
    Generates and returns a TensorFlow graph containing postprocessing
    operations. By convention, this graph contains one or more input
    placeholders that correspond to output ops by the same name in
    the main graph.

    For each output in the original graph that needs postprocessing,
    the preprocessing graph should contain an input placeholder with the same
    name and a second op named "<name of output>_postprocessed",
    where `<name of output>` is the name of the original output op.
    """



    _HASH_TABLE_INIT_OP_NAME = "hash_table_init"

    label_file = util.fetch_or_use_cached(_CACHE_DIR, "labels.pbtext",
                                          _LABEL_MAP_URL)

    # Category mapping comes in pbtext format. Translate to the format that
    # TensorFlow's hash table initializers expect (key and value tensors).
    with open(label_file, "r") as f:
      raw_data = f.read()
    # Parse directly instead of going through the protobuf API dance.
    records = raw_data.split("}")
    records = records[0:-1]  # Remove empty record at end
    records = [r.replace("\n", "") for r in records] # Strip newlines
    regex = re.compile(r"item {  name: \".+\"  id: (.+)  display_name: \"(.+)\"")
    keys = []
    values = []
    for r in records:
      match = regex.match(r)
      keys.append(int(match.group(1)))
      values.append(match.group(2))

    result_decode_g = tf.Graph()
    with result_decode_g.as_default():
      # The original graph produces floating-point output for detection class,
      # even though the output is always an integer.
      float_class = tf.placeholder(tf.float32, shape=[None],
                                   name="detection_classes")
      int_class = tf.cast(float_class, tf.int32)
      key_tensor = tf.constant(keys, dtype=tf.int32)
      value_tensor = tf.constant(values)
      table_init = tf.contrib.lookup.KeyValueTensorInitializer(
        key_tensor,
        value_tensor,
        name=_HASH_TABLE_INIT_OP_NAME)
      hash_table = tf.contrib.lookup.HashTable(
        table_init,
        default_value="Unknown"
      )
      _ = hash_table.lookup(int_class, name="detection_classes_postprocessed")
    return result_decode_g


################################################################################
# CALLBACKS FOR PRE/POST-PROCESSING


# BEGIN MARKER FOR CODE GENERATOR -- DO NOT DELETE
class ObjectDetectorHandlers(PrePost):

  def pre_process(self, request):
    # type: (InferenceRequest) -> None
    """
    Preprocessing callback. Maps an input JSON request to a request that can
    be directly passed to the graph in the underlying SavedModel.

    Args:
      request: InferenceRequest object whose "raw_inputs" field is populated
        with key-value pairs.
        Implementations of this method should populate the
        "processed_inputs" field of `request`.
    """
    # raw_inputs keys used:
    # image: Raw image data as Python bytes
    #
    # processed_inputs keys produced:
    # image_tensor: Image data as a Python bytes
    request.processed_inputs["image_tensor"] = request.raw_inputs["image"]

  def post_process(self, request):
    # type: (InferenceRequest) -> None
    """
    Postprocessing callback. Maps the raw output of the TensorFlow model to a
    format that can be translated to JSON for transmission over the wire.

    Args:
      request: InferenceRequest object whose "raw_outputs" field is populated
        with key-value pairs.
        Implementations of this method should populate the
        "processed_outputs" field of `request`.
    """
    # raw_inputs keys used:
    # threshold: Numeric detection threshold, 0.0 - 1.0
    #
    # raw_outputs keys used:
    # detection_boxes: Bounding boxes as float32 tensors
    # detection_classes: String class labels for bounding boxes
    # detection_scores: float32 detection scores, 0.0 - 1.0
    # num_detections: Integer encoded as a float32; how many entries of the
    #                 other three outputs contain data instead of garbage.
    #
    # processed_outputs keys produced:
    # status: String result status. "ok" if everything went ok, error message
    # otherwise.
    # predictions: Array of detected objects in the format:
    #   "predictions": [
    #     {
    #       "label": "boat",
    #       "probability": 0.8920367360115051,
    #       "detection_box": [
    #         0.5134784579277039,
    #         0.5150489211082458,
    #         0.7650228142738342,
    #         1
    #       ]
    #     }
    #   ]
    boxes = request.raw_outputs["detection_boxes"]
    classes = request.raw_outputs["detection_classes"]
    scores = request.raw_outputs["detection_scores"]
    num_detections = int(request.raw_outputs["num_detections"])
    predictions = []
    for i in range(num_detections):
      probability = float(scores[0, i])
      if probability > request.raw_inputs["threshold"]:
        classes_value = classes[0, i]
        if isinstance(classes_value, bytes):
          classes_value = classes_value.decode("utf-8")
        predictions.append({
          "label": classes_value,
          "probability": probability,
          "detection_box": boxes[0, i].tolist()
        })
    request.processed_outputs["status"] = "ok"
    request.processed_outputs["predictions"] = predictions
    print("Predictions: {}".format(predictions))

  def error_post_process(self, request, error_message):
    # type: (InferenceRequest, str) -> None
    """
    Postprocessing callback that is invoked in the case of an error during
    inference.

    Args:
      request: InferenceRequest object associated with the failed request
      error_message: String that describes what went wrong during inference
    """
    # processed_outputs keys produced:
    # status: String result status. "ok" if everything went ok, error message
    # otherwise.
    request.processed_outputs["status"] = error_message

# END MARKER FOR CODE GENERATOR -- DO NOT DELETE
