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
import os
import tensorflow as tf



def main():
  """
  Generate a deployable WML function from the handlers in the file
  "handlers.py" in this directory.

  Output goes to "wml_function.py"
  """
  func_body = util.generate_wml_function(handlers.ObjectDetectorHandlers)
  print(func_body)


if __name__ == "__main__":
  main()

