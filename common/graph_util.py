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
"""Utilities for manipulating graphs.

Much of the code here should be merged into graph_def_editor once it's
stable.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

# Local imports
import graph_def_editor as gde


def graph_has_op(g: tf.Graph, op_name: str):
  """
  A method that really ought to be part of `tf.Graph`. Returns true of the
  indicated graph has an op by the indicated name.
  """
  all_ops_in_graph = g.get_operations()
  names_of_all_ops_in_graph = [o.name for o in all_ops_in_graph]
  return op_name in names_of_all_ops_in_graph


def add_preprocessing(g, preproc_g):
  # type: (gde.Graph, gde.Graph) -> None
  """
  Add preprocessing ops to a graph.

  Replaces one or more input `Placeholders` in the target graph with
  subgraphs that preprocess the input values prior to feeding them into the
  original graph.

  After performing this rewrite, the inputs of the resulting graph may have a
  different shape and dtype than before, but they will have the same names.

  Args:
    g: `gde.Graph` to which preprocessing should be added. *Modified in place.*
    preproc_g: `gde.Graph` containing the preprocessing ops to add.
      For each placeholder in `g` that needs preprocessing, `preproc_g`
      should contain a placeholder with the same name and a second op named
      "<name of placeholder>_preprocessed", where `<name of placeholder>` is
      the name of the Placeholder op.
  """
  placeholders = gde.filter_ops_by_optype(preproc_g, "Placeholder")

  def preproc_name(placeholder_name):
    return placeholder_name + "_preprocessed"

  def orig_name(placeholder_name):
    return "__original__" + placeholder_name

  # Validate before modifying the graph
  for p in placeholders:
    if not g.contains_node(p.name):
      raise ValueError("Preprocessing graph contains a Placeholder called "
                       "'{}', but target graph does not have an input "
                       "Placeholder by that name."
                       "".format(p.name))
    if not preproc_g.contains_node(preproc_name(p.name)):
      raise ValueError("Preprocessing graph contains a Placeholder called "
                       "'{}', but it does not have an output node called '{}' "
                       "to produce the preprocessed version of that input."
                       "".format(p.name, preproc_name(p.name)))

  # Rename all the target placeholders so we can bulk-copy the preprocessing
  # graph.
  for p in placeholders:
    g.rename_node(p.name, orig_name(p.name))

  # Now it should be safe to copy the preprocessing graph into the original
  # graph.
  gde.copy(preproc_g, g)

  for p in placeholders:
    preproc_p = g.get_node_by_name(preproc_name(p.name))
    orig_p = g.get_node_by_name(orig_name(p.name))

    # Reroute all connections from original placeholder to go to the
    # corresponding output of the preprocessing graph.
    gde.reroute_ts(preproc_p.output(0), orig_p.output(0))

    # Get rid of the original placeholder
    g.remove_node_by_name(orig_p.name)


def add_postprocessing(g, postproc_g):
  # type: (gde.Graph, gde.Graph) -> None
  """
  Add postprocessing ops to a graph.

  The postprocessing ops can replace one or more output operations of the
  original graph with a series of operations that apply additional
  transformations to the output and return the result of the transformations.

  After performing this rewrite, the outputs of the resulting graph may have a
  different shape and dtype than before, but they will have the same names.

  Args:
    g: `gde.Graph` to which postprocessing should be added. *Modified in place.*
    postproc_g: `gde.Graph` containing the postprocessing ops to add.
      For each op in `g` that needs postprocessing, `postproc_g`
      should contain a placeholder with the same name and a second op named
      "<name of output>_postprocessed", where `<name of output>` is
      the name of the original op.
  """
  placeholders = gde.filter_ops_by_optype(postproc_g, "Placeholder")

  def postproc_name(placeholder_name):
    return placeholder_name + "_postprocessed"

  def orig_name(placeholder_name):
    return "__original__" + placeholder_name

  # Validate before modifying the graph
  for p in placeholders:
    if not g.contains_node(p.name):
      raise ValueError("Postprocessing graph contains a Placeholder called "
                       "'{}', but target graph does not have an op by that "
                       "name".format(p.name))
    if 1 != len(g.get_node_by_name(p.name).outputs):
      raise ValueError("Output node '{}' of target graph has {} output "
                       "tensors. Only one output is supported."
                       "".format(p.name,
                                 len(g.get_node_by_name(p.name).outputs)))
    if not postproc_g.contains_node(postproc_name(p.name)):
      raise ValueError("Postprocessing graph contains a Placeholder called "
                       "'{}', but it does not have a node called '{}' "
                       "to produce the postprocessed version of that output."
                       "".format(p.name, postproc_name(p.name)))

  # Rename all the original output ops so we can bulk-copy the preprocessing
  # graph.
  for p in placeholders:
    g.rename_node(p.name, orig_name(p.name))

  # Now it should be safe to copy the preprocessing graph into the original
  # graph.
  gde.copy(postproc_g, g)

  for p in placeholders:
    postproc_input_p = g.get_node_by_name(p.name)
    orig_output_node = g.get_node_by_name(orig_name(p.name))

    # Reroute all connections from original placeholder to go to the
    # corresponding output of the original graph.
    gde.reroute_ts(orig_output_node.output(0), postproc_input_p.output(0))

    # Get rid of the placeholder
    g.remove_node_by_name(postproc_input_p.name)

    # Rename the postprocessed output to the name of the original output
    g.rename_node(postproc_name(p.name), p.name)
