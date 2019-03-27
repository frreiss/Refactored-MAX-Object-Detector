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


# Panda pic from Wikimedia; also used in
# https://github.com/tensorflow/models/blob/master/research/slim/nets ...
#   ... /mobilenet/mobilenet_example.ipynb
_PANDA_PIC_URL = ("https://upload.wikimedia.org/wikipedia/commons/f/fe/"
                  "Giant_Panda_in_Beijing_Zoo_1.JPG")

_TMP_DIR = "./temp"

_SAVED_MODEL_DIR = "./saved_model"


def main():
  """
  Connect to a copy of the model deployed via the deploy_wml.py script,
  generate a web service request, pass that request through the model,
  and print the result.

  Before running this script, you need to perform the following manual steps:
  * Perform the manual steps outlined in the deploy_wml.py script.
  * Run the deploy_wml.py script
  * Enter the deployment URL that the deploy_wml.py script prints out into
    the local file `ibm_cloud_credentials.json` under the key
    "WML_function_url".
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
  request_json = {"values": {
    "image": base64.standard_b64encode(image_data).decode("utf-8"),
    "threshold": thresh
  }}

  # Connect to Watson Machine Learning Python API
  with open("./ibm_cloud_credentials.json") as f:
    creds_json = json.load(f)
  _WML_CREDENTIALS = creds_json["WML_credentials"]
  _WML_FUNCTION_URL = creds_json["WML_function_url"]
  client = WatsonMachineLearningAPIClient(_WML_CREDENTIALS)

  response = client.deployments.score(_WML_FUNCTION_URL, request_json)
  print("Response: {}".format(response))


if __name__ == "__main__":
  main()
