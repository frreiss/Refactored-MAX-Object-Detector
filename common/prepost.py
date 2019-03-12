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
