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
from ibm_botocore.client import Config
from ibm_botocore.exceptions import ClientError
import ibm_botocore

ibm_boto3.set_stream_logger('')
from ibm_botocore import history
history.get_global_history_recorder().enable()


# System imports
import os
import json


################################################################################
# CONSTANTS

# Authorization endpoint used in the example code at
# https://console.bluemix.net/docs/services/cloud-object-storage/libraries/python.html#client-credentials
# This may or may not be the only endpoint. This may or not work with all
# object storage instances.
_COS_AUTH_ENDPOINT = "https://iam.cloud.ibm.com/oidc/token"

# Name of the object storage bucket we use to hold the "input" to the
# dummy "training script"
_INPUT_BUCKET = "max-input-bucket3"

################################################################################
# SUBROUTINES


def _empty_cos_bucket(cos, bucket_name, location_constraint):
  # type: (Any, str, str) -> None
  """
  Get a COS bucket into a state of being empty. Removes all contents if the
  bucket has any. Creates the bucket if necessary.

  Args:
    cos: ibm_boto3 COS client instance
    bucket_name: Name of the bucket to wipe
    location_constraint: What "location constraint" code to use when creating
      the bucket if we need to create the bucket. Ignored if the bucket
      already exists.
  """
  # Step 1: Remove all existing entries in the bucket
  files = cos.Bucket(bucket_name).objects.all()
  #try:
  for file in files:
    print("Got file:{}".format(file))
    try:
       cos.Object(bucket_name, file).delete()
       print("Deleted file {}".format(file))
    except ClientError as ce:
       print("CLIENT ERROR while deleting: {}".format(ce))
  # except ClientError as ce:
  #   # Assume that this error is "bucket not found" and keep going.
  #   print("IGNORING CLIENT ERROR: {}".format(ce))
  #   pass

  # Step 2: Create the bucket
  #try:
  cos.Bucket(bucket_name).create(
    #CreateBucketConfiguration={
    #  "LocationConstraint": location_constraint
    # }
  )
  # except ClientError as be:
  #   print("CLIENT ERROR: {0}\n".format(be))
  # except Exception as e:
  #   print("Unable to create bucket: {0}".format(e))




################################################################################
# BEGIN SCRIPT
def main():
  """
  Script to deploy this model to Watson Machine Learning.

  Before running this script, you must perform the following manual steps:


  * Create a file `ibm_cloud_credentials.json` in this directory.
    Initialize the file with an empty JSON record, i.e. "{ }".
  * Create a Cloud Object Storage instance.
  * Go to the COS web UI and create a set of credentials with write
    permissions on your COS instance. Paste the JSON version of the credentials
    into `ibm_cloud_credentials.json` under the key "COS_credentials"
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
    under the key "COS_location_constraint"

  """
  # STEP 1: Read IBM Cloud authentication data from the user's local JSON
  # file.
  with open("./ibm_cloud_credentials.json") as f:
    creds_json = json.load(f)

  print("creds_json is:\n{}".format(creds_json))

  _COS_ENDPOINT = creds_json["COS_endpoint"]
  _COS_API_KEY_ID = creds_json["COS_credentials"]["apikey"]
  _COS_RESOURCE_CRN = creds_json["COS_credentials"]["resource_instance_id"]

  # The following is not currently used
  _COS_BUCKET_LOCATION = creds_json["COS_location_constraint"]
  #
  #
  #
  # # STEP 1: Create a bucket on Cloud Object Storage to hold the SavedModel
  # cos = ibm_boto3.resource("s3",
  #                          ibm_api_key_id=_COS_API_KEY,
  #                          config=Config(signature_version="oauth"),
  #                          endpoint_url=_COS_ENDPOINT
  #                          )

  # _COS_ENDPOINT = "https://s3.us-south.cloud-object-storage.appdomain.cloud"
  # _COS_API_KEY_ID = "uY6XkaaWUFH72isqz_-o7oGIBeWjwkbQqCJ7sN0BSHeG"
  # _COS_AUTH_ENDPOINT = "https://iam.cloud.ibm.com/oidc/token"
  # _COS_RESOURCE_CRN = "crn:v1:bluemix:public:cloud-object-storage:global:a" \
  #                    "/5002ac6ab716056af8856f864d42d6ed:52e4d0fb-7df4-4f58
  #                    -abdf-2466b77ea82e::"
  # _COS_BUCKET_LOCATION = "us-standard"

  # Create resource
  cos = ibm_boto3.resource("s3",
                           ibm_api_key_id=_COS_API_KEY_ID,
                           ibm_service_instance_id=_COS_RESOURCE_CRN,
                           ibm_auth_endpoint=_COS_AUTH_ENDPOINT,
                           config=Config(signature_version="oauth"),
                           endpoint_url=_COS_ENDPOINT
                           )

  _empty_cos_bucket(cos, _INPUT_BUCKET, _COS_BUCKET_LOCATION)

  # STEP 2: Convert the SavedModel directory to a tarball. WML expects a
  # tarball.

  # STEP 3: Upload the SavedModel tarball to the COS bucket.

  # func_body = util.generate_wml_function(handlers.ObjectDetectorHandlers)
  # print(func_body)
  print("Done.")


if __name__ == "__main__":
  main()

