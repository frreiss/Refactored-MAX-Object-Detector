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

import json
import tensorflow as tf


# BEGIN MARKER FOR CODE GENERATOR -- DO NOT DELETE
class InferenceRequest(object):
  """
  Class for representing in-flight inference reqeusts as they go through
  preprocessing, inference, and postprocessing.
  """
  def __init__(self):
    """
    Create an empty request object.
    """
    self._raw_inputs = {}  # type: Dict[str, Any]
    self._processed_inputs = {}  # type: Dict[str, Any]
    self._raw_outputs = {}  # type: Dict[str, Any]
    self._processed_outputs = {}  # type: Dict[str, Any]

  @property
  def raw_inputs(self):
    # type: () -> Dict[str, Any]
    """
    The raw inputs to the inference request, expressed as a set of key-value
    pairs. Keys are strings. Values are raw data types as they came in on the
    web service request.
    """
    return self._raw_inputs

  @property
  def processed_inputs(self):
    # type: () -> Dict[str, Any]
    """
    Preprocessed inputs to the inference request as key-value pairs. Keys are
    strings that correspond to input argument names for the SavedModel.
    Values are Python objects suitable for feeding to the indicated arguments
    when invoking a TensorFlow graph.
    """
    return self._processed_inputs

  @property
  def raw_outputs(self):
    # type: () -> Dict[str, Any]
    """
    The raw inputs to the inference request, expressed as a set of key-value
    pairs.
    Keys are strings that correspond to outputs of the SavedModel.
    Values are Python objects as returned by TensorFlow.
    """
    return self._raw_outputs

  @property
  def processed_outputs(self):
    # type: () -> Dict[str, Any]
    """
    Postprocess outputs of the inference request as key-value pairs.
    Keys are strings. Values are JSON data types translated to the
    closest equivalent Python types by `json.loads()`
    """
    return self._processed_outputs

  def json_result(self):
    # type: () -> str
    """
    Generate a human-readable JSON string version of the result of this request.
    """
    return json.dumps(self.processed_outputs, indent=4)
# END MARKER FOR CODE GENERATOR -- DO NOT DELETE


# We keep this function separate from the class so that the class doesn't
# depend on TensorFlow
def pass_to_local_tf(
        request, # type: InferenceRequest
        sess, # type: tf.Session
        graph, # type: tf.Graph
        signature # type: tf.SignatureDef
  ):
  # type: (...) -> Dict[str, Any]
  """
  Pass the processed inputs of this request to a local TensorFlow graph,
  emulating the way that TensorFlow Serving would handle the request.
  Populates `request.raw_outputs` with the results.

  Args:
    request: Request to pass to local TensorFlow
    sess: TensorFlow session in which the graph lives
    graph: Graph that has been initialized with the model that this
    inference request targets
    signature: "Method" signature from the SavedModel
  """
  input_dict = {}
  for key in signature.inputs:
    tensor_name = signature.inputs[key].name
    input_dict[tensor_name] = self.processed_inputs[key]
  fetch_tensor_names = []
  fetch_output_names = []
  for key in signature.outputs:
    tensor_name = signature.outputs[key].name
    fetch_tensor_names.append(tensor_name)
    fetch_output_names.append(key)
  results = sess.run(fetch_tensor_names, feed_dict=input_dict)
  for i in range(len(fetch_output_names)):
    output_name = fetch_output_names[i]
    self.raw_outputs[output_name] = results[i]




