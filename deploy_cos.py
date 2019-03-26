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

# IBM Cloud imports
import ibm_boto3
import ibm_boto3.s3.transfer
from ibm_botocore.client import Config
from ibm_botocore.exceptions import ClientError

from watson_machine_learning_client import WatsonMachineLearningAPIClient

# Uncomment to enable tracing of object storage REST calls.
# ibm_boto3.set_stream_logger('')

# Leftover from an attempt to enable further logging. Didn't seem to work
# back then.
# from ibm_botocore import history
# history.get_global_history_recorder().enable()


# System imports
import os
import json
import subprocess
import sys

################################################################################
# CONSTANTS

# Authorization endpoint used in the example code at
# https://console.bluemix.net/docs/services/cloud-object-storage/libraries/
# python.html#client-credentials
# This may or may not be the only endpoint. This may or not work with all
# object storage instances.
_COS_AUTH_ENDPOINT = "https://iam.cloud.ibm.com/oidc/token"

# Name of the object storage bucket we use to hold the model's artifacts.
# Currently we assume one bucket per model.
_MODEL_BUCKET = "MAX-Object-Detector-Bucket"

_SAVED_MODEL_DIR = "./saved_model"
_SAVED_MODEL_TARBALL = _SAVED_MODEL_DIR + ".tar.gz"
_SAVED_MODEL_TARBALL_PATH_IN_COS = "saved_model.tar.gz"

# Preshrunk configuration object handle for ibm_boto3 block transfers
_ONE_MEGABYTE = 1024 * 1024
_COS_TRANSFER_BLOCK_SIZE = 16 * _ONE_MEGABYTE
_COS_TRANSFER_MULTIPART_THRESHOLD = 128 * _ONE_MEGABYTE
_COS_TRANSFER_CONFIG_OBJECT = ibm_boto3.s3.transfer.TransferConfig(
  multipart_threshold=_COS_TRANSFER_MULTIPART_THRESHOLD,
  multipart_chunksize=_COS_TRANSFER_BLOCK_SIZE
)

# Various metadata fields that you need to pass to the WML service when
# deploying a model.
_WML_META_AUTHOR_NAME = "CODAIT"
_WML_META_NAME = "MAX Object Detector"
_WML_META_DESCRIPTION = "Test MAX Models"
_WML_META_FRAMEWORK_NAME = "tensorflow"
_WML_META_FRAMEWORK_VERSION = "1.6"
_WML_META_RUNTIME_NAME = "python"
_WML_META_RUNTIME_VERSION = "3.6"
#_WML_META_RUNTIME_NAME = "python"
#_WML_META_RUNTIME_VERSION = "3.6"
#_WML_META_FRAMEWORK_LIBRARIES = [{'name':'keras', 'version': '2.1.3'}]


################################################################################
# SUBROUTINES


def _empty_cos_bucket(cos, bucket_name, location_constraint):
  # type: (Any, str, str) -> None
  """
  Get a COS bucket into a state of being empty. Removes all contents if the
  bucket has any. Creates the bucket if necessary.

  Args:
    cos: initialized and connected ibm_boto3 COS client instance
    bucket_name: Name of the bucket to wipe
    location_constraint: What "location constraint" code to use when creating
      the bucket if we need to create the bucket. Ignored if the bucket
      already exists.
  """
  # Step 1: Remove any existing objects in the bucket
  # Note that the Bucket.objects.all() returns a generator. No REST calls are
  # made on the following line.
  files = cos.Bucket(bucket_name).objects.all()
  try:
    for file in files:
      try:
        cos.Object(bucket_name, file).delete()
        print("Deleted file {}".format(file))
      except ClientError as ce:
        print("CLIENT ERROR while deleting: {}".format(ce))
  except ClientError as ce:
    # Assume that this error is "bucket not found" and keep going.
    print("IGNORING CLIENT ERROR: {}".format(ce))
    pass

  # Step 2: Create the bucket (noop if bucket already exists)
  try:
    cos.Bucket(bucket_name).create(
      CreateBucketConfiguration={
        "LocationConstraint": location_constraint
      }
    )
  except ClientError as be:
    # Flow through only if the call failed because the bucket already exists.
    if "BucketAlreadyExists" not in str(be):
      raise be


def _cp_to_cos(cos, local_file, bucket_name, item_name, replace=False):
  """
  Magic formula for "cp <local_file> <bucket_name>/<item_name>". Roughly
  equivalent to installing the AWS command line tools, redoing all the work
  you've done to set up IBM Cloud Service credentials in order get the AWS
  command line tools configured with working HMAC keys, then running
  `aws s3 cp <local_file> s3://<server>/<bucket_name>/<item_name>`.

  See https://cloud.ibm.com/docs/services/cloud-object-storage/libraries?
  topic=cloud-object-storage-using-python#upload-binary-file-preferred-method-
  for more information.

  Args:
    cos: initialized and connected ibm_boto3 COS client instance
    local_file: Single local file to copy to your Cloud Object Storage bucket
    bucket_name: Name of the target bucket
    item_name: Path within the bucket at which the object should be copied
    replace: If True, overwrite any existing object at the target location. If
      False, raise an exception if the target object already exists.
  """
  try:
    with open(local_file, "rb") as file_data:
      cos.Object(bucket_name, item_name).upload_fileobj(
        Fileobj=file_data,
        Config=_COS_TRANSFER_CONFIG_OBJECT
      )
  except ClientError as be:
    print("CLIENT ERROR: {0}\n".format(be))
  except Exception as e:
    print("Unable to complete multi-part upload: {0}".format(e))


################################################################################
# BEGIN SCRIPT
def main():
  """
  Script to deploy the larger parts of this model to Cloud Object Storage.

  Before running this script, you'll need to perform the following manual steps:
  * Create a file `ibm_cloud_credentials.json` in this directory, if such a
    file doesn't already exist.
    Initialize the file with an empty JSON record, i.e. "{ }".
  * Create a Cloud Object Storage instance.
  * Go to the COS web UI and create a set of credentials with write
    permissions on your COS instance. Paste the JSON version of the credentials
    into `ibm_cloud_credentials.json` under the key "COS_credentials".
  * Figure out an endpoint that your COS instance can talk to.
    Go back to the web UI for COS and click on "Endpoints". Take one of the
    endpoint names, prepend it with "https://", and store the resulting URL
    under the key "COS_endpoint" in `ibm_cloud_credentials.json`
  * Figure out what "location constraint" works for your COS bucket. Today
    there is a list of potential values at
  https://console.bluemix.net/docs/infrastructure/cloud-object-storage
  -infrastructure/buckets.html#create-a-bucket
    (though this part of the docs has a habit of moving around).
    Enter your location constraint string into `ibm_cloud_credentials.json`
    under the key "COS_location_constraint".
  """
  # STEP 1: Read IBM Cloud authentication data from the user's local JSON
  # file.
  with open("./ibm_cloud_credentials.json") as f:
    creds_json = json.load(f)

  print("creds_json is:\n{}".format(creds_json))

  _COS_ENDPOINT = creds_json["COS_endpoint"]
  _COS_API_KEY_ID = creds_json["COS_credentials"]["apikey"]
  _COS_RESOURCE_CRN = creds_json["COS_credentials"]["resource_instance_id"]
  _COS_LOCATION_CONSTRAINT = creds_json["COS_location_constraint"]

  _WML_CREDENTIALS = creds_json["WML_credentials"]
  _WML_USER_NAME = creds_json["WML_credentials"]["username"]
  _WML_PASSWORD = creds_json["WML_credentials"]["password"]
  _WML_INSTANCE = creds_json["WML_credentials"]["instance_id"]
  _WML_URL = creds_json["WML_credentials"]["url"]

  # STEP 2: Create a bucket on Cloud Object Storage to hold the SavedModel
  cos = ibm_boto3.resource("s3",
                           ibm_api_key_id=_COS_API_KEY_ID,
                           ibm_service_instance_id=_COS_RESOURCE_CRN,
                           ibm_auth_endpoint=_COS_AUTH_ENDPOINT,
                           config=Config(signature_version="oauth"),
                           endpoint_url=_COS_ENDPOINT
                           )

  _empty_cos_bucket(cos, _MODEL_BUCKET, _COS_LOCATION_CONSTRAINT)

  # STEP 3: Convert the SavedModel directory to a tarball.
  if os.path.exists(_SAVED_MODEL_TARBALL):
    os.remove(_SAVED_MODEL_TARBALL)
  subprocess.run(["tar", "--create", "--gzip", "--verbose",
                  "--directory={}".format(_SAVED_MODEL_DIR),
                  "--file={}".format(_SAVED_MODEL_TARBALL),
                  "saved_model.pb"])

  # STEP 4: Upload the SavedModel tarball to the COS bucket.
  _cp_to_cos(cos, _SAVED_MODEL_TARBALL, _MODEL_BUCKET,
             _SAVED_MODEL_TARBALL_PATH_IN_COS, replace=True)

  print("Done.")


if __name__ == "__main__":
  main()
