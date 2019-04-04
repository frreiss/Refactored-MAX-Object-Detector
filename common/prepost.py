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

import tensorflow as tf

# BEGIN MARKER FOR CODE GENERATOR -- DO NOT REMOVE
class PrePost(object):
  """
  Base class for pre/post-processing callbacks in Python.
  """

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
    raise NotImplementedError()

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
    raise NotImplementedError()

  def error_post_process(self, request, error_message):
    # type: (InferenceRequest, str) -> None
    """
    Postprocessing callback that is invoked in the case of an error during
    inference.

    Args:
      request: InferenceRequest object associated with the failed request
      error_message: String that describes what went wrong during inference
    """
    raise NotImplementedError()
# END MARKER FOR CODE GENERATOR -- DO NOT REMOVE


class GraphGen(object):
  """
  Base class for graph generation callbacks in Python.
  """

  def frozen_graph(self):
    # type: () -> tf.GraphDef
    """
    Generates and returns the core TensorFlow graph for the model as a frozen
    (i.e. all variables converted to constants) GraphDef protocol buffer
    message.
    """
    raise NotImplementedError()

  def input_node_names(self):
    # type: () -> List[str]
    """
    Returns a list of the names of Placeholder ops (AKA nodes) in the graph
    returned by `frozen_graph` that are required inputs for inference.
    """
    raise NotImplementedError()

  def output_node_names(self):
    """
    Returns a list of the names of  ops (AKA nodes) in the graph returned by
    `frozen_graph` that produce output values for inference requests.
    """
    raise NotImplementedError()

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
    raise NotImplementedError()

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
    raise NotImplementedError()

