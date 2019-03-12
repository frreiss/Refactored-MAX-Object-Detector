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

from common.prepost import PrePost
from common.inference_request import InferenceRequest


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
        predictions.append({
          "label": classes[0, i].decode("utf-8"),
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
