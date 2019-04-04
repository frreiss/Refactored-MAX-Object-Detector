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

"""
Script that builds the TensorFlow graph for the Model Asset Exchange's Object 
Detector model.

This script starts with the pre-trained object detection model from the
TensorFlow Models repository; see https://github.com/tensorflow/models/
blob/master/research/object_detection/g3doc/detection_model_zoo.md.
Specifically, we use the object detector trained on the COCO dataset with a
MobileNetV1 architecture.

The original model takes as input batches of equal-sized images, represented
as a single dense numpy array of binary pixel data.  The output of the
original model represents the object type as an integer. This script grafts on
pre- and post-processing ops to make the input and output format more amenable
to use in applications. After these ops are added, the resulting graph takes a
single image file as an input and produces string-valued object labels.

To run this script from the root of the project, type:
   env/bin/python build_graph.py

The output SavedModel file will be written to ./saved_model

The script also creates temporary files in ./temp, including dumps of the 
graph at various phases of processing.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import graph_def_editor as gde
import shutil
from typing import List
import tempfile
from tensorflow.tools import graph_transforms
import textwrap

# Local imports
from common import graph_util, util, prepost
import handlers

FLAGS = tf.flags.FLAGS


def _indent(s):
  return textwrap.indent(str(s), "    ")


###############################################################################
# CONSTANTS
_HASH_TABLE_INIT_OP_NAME = "hash_table_init"
_PYTHON_SAVED_MODEL_DIR = "./saved_model"
_JS_SAVED_MODEL_DIR = "./saved_model_js"


def _apply_graph_transform_tool_rewrites(g: gde.Graph,
                                         input_node_names: List[str],
                                         output_node_names: List[str]) \
        -> tf.GraphDef:
  """
  Use the [Graph Transform Tool](
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/
  graph_transforms/README.md)
  to perform a series of pre-deployment rewrites.

  Args:
     g: GDE representation of the core graph.
     input_node_names: Names of placeholder nodes that are used as inputs to
       the graph for inference. Placeholders NOT on this list will be
       considered dead code.
     output_node_names: Names of nodes that produce tensors that are outputs
       of the graph for inference purposes. Nodes not necessary to produce
       these tensors will be considered dead code.

  Returns: GraphDef representation of rewritten graph.
  """
  # Invoke the Graph Transform Tool using the undocumented Python APIs under
  # tensorflow.tools.graph_transforms
  after_tf_rewrites_graph_def = graph_transforms.TransformGraph(
    g.to_graph_def(),
    inputs=input_node_names,
    outputs=output_node_names,
    # Use the set of transforms recommended in the README under "Optimizing
    # for Deployment"
    transforms=['strip_unused_nodes(type=float, shape="1,299,299,3")',
                'remove_nodes(op=Identity, op=CheckNumerics)',
                'fold_constants(ignore_errors=true)',
                'fold_batch_norms',
                'fold_old_batch_norms']
  )
  return after_tf_rewrites_graph_def


def _apply_generic_deployment_rewrites(graph, graph_gen, temp_dir):
  # type: (gde.Graph, prepost.GraphGen, str) -> gde.Graph
  """
  Common code to apply general-purpose graph optimization rewrites that
  remove unnecessary portions of the graph in preparation for inference.

  Args:
    graph: `gde.Graph` object containing the graph after adding any pre/post
      subgraphs
    graph_gen: Graph generation callbacks object for the current model
    temp_dir: Location where this method should write out temp files

  Returns the modified graph as a `gde.Graph` object
  """
  # Now run through some of TensorFlow's built-in graph rewrites.
  output_nodes = graph_gen.output_node_names()

  # Need to treat initializer nodes, if present, as output nodes for the
  # purposes of the Graph Transform Tool
  if graph.contains_node(_HASH_TABLE_INIT_OP_NAME):
    output_nodes.append(_HASH_TABLE_INIT_OP_NAME)

  after_tf_rewrites_graph_def = _apply_graph_transform_tool_rewrites(
    graph, graph_gen.input_node_names(), output_nodes)
  util.protobuf_to_file(after_tf_rewrites_graph_def,
                        temp_dir + "/after_tf_rewrites_graph.pbtext",
                        "Graph after built-in TensorFlow rewrites")

  print("    Number of ops after built-in rewrites: {}".format(len(
    after_tf_rewrites_graph_def.node)))

  # Now run the GraphDef editor's graph prep rewrites
  g = gde.Graph(after_tf_rewrites_graph_def)
  gde.rewrite.fold_batch_norms(g)
  gde.rewrite.fold_old_batch_norms(g)
  gde.rewrite.fold_batch_norms_up(g)
  after_gde_graph_def = g.to_graph_def(add_shapes=True)
  util.protobuf_to_file(after_gde_graph_def,
                        temp_dir + "/after_gde_rewrites_graph.pbtext",
                        "Graph after fold_batch_norms_up() rewrite")

  print("         Number of ops after GDE rewrites: {}".format(len(
    after_gde_graph_def.node)))
  return g


def _make_python_deployable_graph(frozen_graph_def, graph_gen,
                                  temp_dir, saved_model_location):
  # type: (tf.GraphDef, prepost.GraphGen, str, str) -> None
  """
  Prepare a SavedModel directory with a graph that is deployable via the
  Python or C++ APIs of TensorFlow.

  Args:
    frozen_graph_def: Base starter graph produced by inference, after turning
      variables to constants but before other rewrites.
    graph_gen: Callback object for current model
    temp_dir: Temporary directory in which to dump intermediate results in
      case they are needed for debugging.
    saved_model_location: Location where the final output SavedModel should go

  Returns:
    A graph that has been optimized and augmented with preprocessing and
    postprocessing ops.
  """
  # Graft the preprocessing and postprocessing graphs onto the beginning and
  # end of the inference graph.
  g = gde.Graph(frozen_graph_def)

  preproc_g = gde.Graph(graph_gen.pre_processing_graph())
  postproc_g = gde.Graph(graph_gen.post_processing_graph())

  graph_util.add_preprocessing(g, preproc_g)
  graph_util.add_postprocessing(g, postproc_g)

  after_add_pre_post_graph_def = g.to_graph_def()
  util.protobuf_to_file(after_add_pre_post_graph_def,
                        temp_dir + "/after_pre_and_post.pbtext",
                        "Graph with pre- and post-processing")

  print("            Number of ops in frozen graph: {}".format(len(
    frozen_graph_def.node)))
  print(" Num. ops after adding pre- and post-proc: {}".format(len(
    after_add_pre_post_graph_def.node)))

  g = _apply_generic_deployment_rewrites(g, graph_gen, temp_dir)

  # Graph preparation complete. Create a SavedModel "file" (actually a
  # directory)
  saved_model_graph = tf.Graph()
  with saved_model_graph.as_default():
    with tf.Session() as sess:
      tf.import_graph_def(g.to_graph_def(), name="")

      # Recreate the hash table initializers collection, which got wiped out
      # when we round-tripped the graph through the GraphDef format.
      hash_table_init_op = saved_model_graph.get_operation_by_name(
        _HASH_TABLE_INIT_OP_NAME)
      saved_model_graph.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS,
                                          hash_table_init_op)

      # simple_save needs pointers to tensors, so pull input and output
      # tensors out of the graph.
      inputs_dict = {
        n: saved_model_graph.get_tensor_by_name(n + ":0")
        for n in graph_gen.input_node_names()
      }
      outputs_dict = {
        n: saved_model_graph.get_tensor_by_name(n + ":0")
        for n in graph_gen.output_node_names()
      }
      if os.path.isdir(saved_model_location):
        shutil.rmtree(saved_model_location)
      tf.saved_model.simple_save(sess,
                                 export_dir=saved_model_location,
                                 inputs=inputs_dict,
                                 outputs=outputs_dict,
                                 legacy_init_op=hash_table_init_op)
  print("SavedModel written to {}".format(saved_model_location))


def _make_javascript_deployable_graph(frozen_graph_def, graph_gen,
                                      temp_dir, saved_model_location):
  # type: (tf.GraphDef, prepost.GraphGen, str, str) -> None
  """
  Prepare a SavedModel directory with a graph that is deployable via
  TensorFlow.js

  Args:
    frozen_graph_def: Base starter graph produced by inference, after turning
      variables to constants but before other rewrites.
    graph_gen: Callbacks for the current model
    temp_dir: Temporary directory in which to dump intermediate results in
      case they are needed for debugging.
    saved_model_location: Location where the final output SavedModel should go

  Returns:
    A graph that has been optimized. No preprocessing or postprocessing ops
    are attached, as the ops we would like to use for those purposes are not
    currently implemented in TensorFlow.js
  """
  g = gde.Graph(frozen_graph_def)

  print("            Number of ops in frozen graph: {}".format(len(
    frozen_graph_def.node)))

  g = _apply_generic_deployment_rewrites(g, graph_gen, temp_dir)

  # Graph preparation complete. Create a SavedModel "file" (actually a
  # directory)
  saved_model_graph = tf.Graph()
  with saved_model_graph.as_default():
    with tf.Session() as sess:
      tf.import_graph_def(g.to_graph_def(), name="")

      # simple_save needs pointers to tensors, so pull input and output
      # tensors out of the graph.
      inputs_dict = {
        n: saved_model_graph.get_tensor_by_name(n + ":0")
        for n in graph_gen.input_node_names()
      }
      outputs_dict = {
        n: saved_model_graph.get_tensor_by_name(n + ":0")
        for n in graph_gen.output_node_names()
      }
      if os.path.isdir(saved_model_location):
        shutil.rmtree(saved_model_location)
      tf.saved_model.simple_save(sess,
                                 export_dir=saved_model_location,
                                 inputs=inputs_dict,
                                 outputs=outputs_dict)
  print("SavedModel written to {}".format(saved_model_location))


def _make_temp_dir():
  """
  Wrapper around tempfile so that we can enable/disable deletion of temp
  directories from a single place
  """
  _DELETE_TEMP_DIRS = False
  if _DELETE_TEMP_DIRS:
    return tempfile.TemporaryDirectory(prefix=".")
  else:
    return tempfile.mkdtemp(prefix=".")


def main(_):
  # We start with a frozen graph for the model. "Frozen" means that all
  # variables have been converted to constants.
  graph_generators = handlers.GraphGenerators()
  frozen_graph_def = graph_generators.frozen_graph()

  util.protobuf_to_file(frozen_graph_def, "frozen_graph.pbtxt",
                        "Frozen graph")

  _make_python_deployable_graph(frozen_graph_def, graph_generators,
                                _make_temp_dir(), _PYTHON_SAVED_MODEL_DIR)
  _make_javascript_deployable_graph(frozen_graph_def, graph_generators,
                                    _make_temp_dir(), _JS_SAVED_MODEL_DIR)


if __name__ == "__main__":
  tf.app.run()
