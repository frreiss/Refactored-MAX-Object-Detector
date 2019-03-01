#! /bin/bash

################################################################################
# env.sh
#
# Set up an Anaconda virtualenv in the directory ./env
#
# Run this script from the root of the project, i.e.
#   ./env.sh
#
# Requires that conda be installed and set up for calling from bash scripts.
#
# Also requires that you set the environment variable CONDA_HOME to the
# location of the root of your anaconda/miniconda distribution.
################################################################################

PYTHON_VERSION=3.6

############################
# HACK ALERT *** HACK ALERT 
# The friendly folks at Anaconda thought it would be a good idea to make the
# "conda" command a shell function. 
# See https://github.com/conda/conda/issues/7126
# The following workaround will probably be fragile.
if [ -z "$CONDA_HOME" ]
then 
    echo "Error: CONDA_HOME not set"
    exit
fi
. ${CONDA_HOME}/etc/profile.d/conda.sh
# END HACK
############################

################################################################################
# Remove any previous outputs of this script

rm -rf ./env
rm -rf ./graph_def_editor

################################################################################
# Create the environment
conda create -y --prefix ./env \
    python=${PYTHON_VERSION} \
    numpy \
    tensorflow \
    jupyterlab \
    pytest \
    keras \
    pillow \
    nomkl

# Install the latest master branch of GDE
#git clone https://github.com/CODAIT/graph_def_editor.git
# Temporary: Use my branch until my latest PR is merged 
git clone https://github.com/frreiss/graph_def_editor.git
cd graph_def_editor
git checkout issue-savedmodel
cd ..
conda activate ./env
pip install ./graph_def_editor/
conda deactivate

# Delay so that the message ends up at the end of script output
sleep 1
echo << EOM
Anaconda virtualenv installed in ./env.
Run \"conda activate ./env\" to use it.
EOM

