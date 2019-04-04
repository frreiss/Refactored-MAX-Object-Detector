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
import json
import os

# IBM imports
from watson_machine_learning_client import WatsonMachineLearningAPIClient

import deployable_function

# Panda pic from Wikimedia; also used in
# https://github.com/tensorflow/models/blob/master/research/slim/nets ...
#   ... /mobilenet/mobilenet_example.ipynb
_PANDA_PIC_URL = ("https://upload.wikimedia.org/wikipedia/commons/f/fe/"
                  "Giant_Panda_in_Beijing_Zoo_1.JPG")

_TMP_DIR = "./temp"

_SAVED_MODEL_DIR = "./saved_model"


def main():
  """
  Connect to a copy of the "core" model deployed via the deploy_wml.py script
  using a local copy of the WML function that was deployed by the
  deploy_wml.py script.

  Before running this script, you need to perform the following manual steps:
  * Perform the manual steps outlined in the deploy_wml.py script.
  * Run the deploy_wml.py script
  * Enter the deployment URL that the deploy_wml.py script prints out into
    the local file `ibm_cloud_credentials.json` under the key
    "WML_function_url".
  * Enter the model ID that the deploy_wml.py script prints out to the local
    file `ibm_cloud_credentials.json` under the key "WML_model_ID". The model
    ID can be found in the part of the output that looks like::
      Model details: {'metadata': {'guid': '<model id>',
    ...or alternately you can set up the CLI with ". bx_env.sh", then run
      bx ml list models
  * Enter the deployment ID that deploy_wml.py prints out to the local
    file `ibm_cloud_credentials.json` under the key "WML_deployment_ID".
    The deployment ID can be found in the part of the script output that
    looks like:
      Deployment details: {'metadata': {'guid': '<deployment id>'
    or you can type
      bx ml list deployments <model id>
    Don't bother typing just "bx ml list deployments". It will return an
    empty set...
  """
  if not os.path.isdir(_TMP_DIR):
    os.mkdir(_TMP_DIR)

  # Prepare a request
  image_path = util.fetch_or_use_cached(_TMP_DIR, "panda.jpg",
                                        _PANDA_PIC_URL)
  with open(image_path, "rb") as f:
    image_data = f.read()
  thresh = 0.7

  # Note that "values" tag at the top level. This tag is a requirement of the
  # WML API standard.
  # Under the "values" tag, you must place a list of tuples. Each tuple must
  # be represented as a JSON list of values. Tensor-valued values must be
  # represented as lists of numbers.
  request_json = {
    "fields": [
      "image",
      "threshold"
    ],
    "values": [
      [
        # TensorFlow only decodes URL-safe base64
        base64.urlsafe_b64encode(image_data).decode("utf-8"),
        thresh
      ]
    ]
  }

  # Write out JSON suitable for passing to "bx ml score"
  with open("./ibm_cloud_credentials.json") as f:
    creds_json = json.load(f)
  _WML_MODEL_ID = creds_json["WML_model_ID"]
  _WML_DEPLOYMENT_ID = creds_json["WML_deployment_ID"]
  cli_json = {
    "modelId": _WML_MODEL_ID,
    "deploymentId": _WML_DEPLOYMENT_ID,
    "payload": request_json
  }
  with open("request.json", "w") as f:
    f.write(json.dumps(cli_json, indent=2))
  print("A copy of the request we're about emulate locally has been saved to "
        "./request.json.  Run\n"
        "   bx ml score request.json\n"
        "to use the WML CLI to run the end-to-end request remotely.")

  func_ptr = deployable_function.deployable_function()
  response = func_ptr(request_json)


if __name__ == "__main__":
  main()
