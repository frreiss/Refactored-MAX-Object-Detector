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
import numpy as np

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

  @raw_inputs.setter
  def raw_inputs(self, value):
    """
    Replace the current value of the raw_inputs property with a *shallow copy*
    of the provided dict.
    """
    self._raw_inputs = value.copy()

  def set_raw_inputs_from_watson_v3(self, request_json):
    # type: (Dict[str, Any]) -> None
    """
    Set the `raw_inputs` property of this request using a JSON request in
    Watson V3 API format, as defined at https://watson-ml-api.mybluemix.net.
    There does not appear to be any formal specification of the input format,
    only the following example:
    ```json
    {
      "fields": [
        "name",
        "age",
        "position"
      ],
      "values": [
        [
          "john",
          33,
          "engineer"
        ],
        [
          "mike",
          23,
          "student"
        ]
      ]
    }
    ```
    This method rigorously enforces the implicit requirements of this example:
    * All fields must be defined in the "fields" tag
    * The "values" tag must contain a list of tuples
    * Each tuple under "values" must have the same number of fields as are
      defined under "fields".
    It's unclear what types are allowed for a field value, so for now we
    allow abitrary JSON for a field value. We also don't attempt to enforce
    that every tuple has the same type in a given field.

    Currently we only support a single input tuple.

    Args:
      request_json: Parsed JSON request in Watson V3 format.
    """
    fields_list = request_json["fields"]
    tuples_list = request_json["values"]
    # TODO: Handle requests with multiple tuples
    if len(tuples_list) > 1:
      raise ValueError("Received {} tuples of values, but current "
                       "implementation can only handle "
                       "1".format(len(tuples_list)))
    first_tuple = tuples_list[0]
    if len(first_tuple) != len(fields_list):
      raise ValueError("Received {} field names and {} field values"
                       "".format(len(fields_list), len(first_tuple)))
    self.raw_inputs.clear()
    for i in range(len(fields_list)):
      field_name = fields_list[i]
      field_value = first_tuple[i]
      self.raw_inputs[field_name] = field_value

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

  @staticmethod
  def value_to_json(v):
    """
    Transform the input value into a Python object will turn back into the
    original value if it is fed through `json.dumps`, then through
    `json.loads`, then through `np.array`; that is:
    ```
      v == np.array(json.loads(json.dumps(value_to_json(v))))
    ```
    """
    if isinstance(v, np.ndarray):
      return v.tolist()
    else:
      return v

  def processed_inputs_as_watson_v3(self):
    # type: () -> Dict[str, Any]
    """
    Convert the `processed_inputs` property of this request to an input
    record in Watson V3 API format, as defined at
    https://watson-ml-api.mybluemix.net.
    The documentation at the above URL doesn't say anything useful about the
    format of model inputs. But there is a bit of documentation squirreled
    away at: https://www.ibm.com/support/knowledgecenter/DSXDOC/analyze-data/
    ml_dlaas_tensorflow_deploy_score.html

    Following the instructions in that second URL, we generate JSON with a
    "keyed_values" field containing key-value pairs.
    """
    key_value_pairs = [
      {
        "key": name,
        "values": InferenceRequest.value_to_json(self.processed_inputs[name])
      }
      for name in self.processed_inputs.keys()
    ]
    return {
      "keyed_values": key_value_pairs
    }

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

  @raw_outputs.setter
  def raw_outputs(self, value):
    """
    Replace the current value of the raw_outputs property with a *shallow copy*
    of the provided dict.
    """
    self._raw_outputs = value.copy()

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




