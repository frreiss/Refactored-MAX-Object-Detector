#! /bin/bash

################################################################################
# bx_env.sh
#
# Set up environment variables for running the bluemix/bx/ibmcloud command
# line utils that start with "bx ml".
#
# Run this script IN THE CURRENT SHELL from the root of the project, i.e.
#   source bx_env.sh
# or
#   . bx_env.sh
#
# Requires that jq and the IBM Cloud command line utilities be installed
# on the local machine, and that your WML credentials are stored in a local
# file ibm_cloud_credentials.json, as described in deploy_wml.py.
################################################################################

# ibm_cloud_credentials.json contains WML credentials as returned by the web
# UI, under the key "WML_credentials"
CREDS="./ibm_cloud_credentials.json"

bx logout

export ML_USERNAME=`cat ${CREDS} | jq --raw-output .WML_credentials.username`
export ML_PASSWORD=`cat ${CREDS} | jq --raw-output .WML_credentials.password`
export ML_INSTANCE=`cat ${CREDS} | jq --raw-output .WML_credentials.instance_id`
export ML_ENV=`cat ${CREDS} | jq --raw-output .WML_credentials.url`

bx login --sso



