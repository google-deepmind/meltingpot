# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A permissive wrapper for a SavedModel."""

import copy
import inspect

from typing import Any, Callable, NamedTuple

from absl import logging
import tensorflow as tf
import tree


class _Function(NamedTuple):
  """Function exposing signature and expected canonical arguments."""
  func: Callable[..., Any]
  signature: inspect.Signature
  structured_specs: Any

  def __call__(self, *args, **kwargs):
    return self.func(*args, **kwargs)

  @property
  def canonical_arguments(self) -> inspect.BoundArguments:
    args, kwargs = copy.deepcopy(self.structured_specs)
    return self.signature.bind(*args, **kwargs)


class PermissiveModel:
  """A permissive wrapper for a SavedModel."""

  # Disable pytype attribute error checks.
  _HAS_DYNAMIC_ATTRIBUTES = True

  def __init__(self, model):
    self.model = model

    self._tables = self.model.function_tables()
    self._initialized_tables = {}

    def build_parameters(params):
      params = [
          inspect.Parameter(str(param[0], "utf-8"), param[1])
          for param in params
      ]
      # Always include a VAR_KEYWORD to capture any extraneous arguments.
      if all([p.kind != inspect.Parameter.VAR_KEYWORD for p in params]):
        params.append(
            inspect.Parameter("__unused_kwargs", inspect.Parameter.VAR_KEYWORD))
      return params

    signatures = self.model.function_signatures()
    if tf.executing_eagerly():
      signatures = tree.map_structure(lambda x: x.numpy(), signatures)
    else:
      with tf.compat.v1.Session() as sess:
        signatures = sess.run(signatures)
    signatures = {
        func_name: inspect.Signature(build_parameters(func_params))
        for func_name, func_params in signatures.items()
    }
    self.signatures = signatures

    # Attach deepfuncs.
    for name in self.signatures.keys():
      setattr(self, name, self._make_permissive_function(name))

  def _maybe_init_tables(self, concrete_func: Any, name: str):
    """Initialise all tables for a function if they are not initialised.

    Some functions rely on hash-tables that must be externally initialized. This
    method will perform a one-time initialisation of the tables. It does so by
    finding the corresponding op that creates the hash-table handles (these will
    be different from the ones observed in the initial deepfuncs), and import
    the corresponding keys and values.

    Args:
      concrete_func: A tf.ConcreteFunction corresponding to a deepfunc.
      name: The name of the deepfunc.
    """
    if name not in self._tables:
      return

    all_nodes = dict(
        main={n.name: n for n in concrete_func.graph.as_graph_def().node})
    for func_def in concrete_func.graph.as_graph_def().library.function:
      all_nodes[func_def.signature.name] = {
          n.name: n for n in func_def.node_def
      }

    for table_name, (table_keys, table_values) in self._tables[name].items():
      table_op = None
      for nodes in all_nodes.values():
        if table_name in nodes:
          if table_op is not None:
            raise ValueError(f"Duplicate table op found for {table_name}")
          table_op = nodes[table_name]

      logging.info("Initialising table for Op `%s`", table_name)
      table_handle_name = table_op.attr["shared_name"].s  # pytype: disable=attribute-error
      table_handle = tf.raw_ops.HashTableV2(
          key_dtype=table_keys.dtype,
          value_dtype=table_values.dtype,
          shared_name=table_handle_name)
      tf.raw_ops.LookupTableImportV2(
          table_handle=table_handle, keys=table_keys, values=table_values)
      self._initialized_tables[name] = self._tables.pop(name)  # Only init once.

  def _make_permissive_function(self, name: str) -> Callable[..., Any]:
    """Create a permissive version of a function in the SavedModel."""
    if name not in self.signatures:
      raise ValueError(f"No function named {name} in SavedModel, "
                       "options are {self.signatures}")

    tf_func = getattr(self.model, name)
    if hasattr(tf_func, "concrete_functions"):
      # tf.RestoredFunction
      concrete_func, = tf_func.concrete_functions  # Expect only one.
    elif hasattr(tf_func, "_list_all_concrete_functions"):
      # tf.Function
      all_concrete_funcs = tf_func._list_all_concrete_functions()  # pylint: disable=protected-access
      all_concrete_signatures = [
          f.structured_input_signature for f in all_concrete_funcs
      ]
      # This is necessary as tf.saved_model.save can force a retrace on
      # tf.Function, resulting in another concrete function with identical
      # signature.
      unique_concrete_signatures = set([
          tuple(tree.flatten_with_path(sig)) for sig in all_concrete_signatures
      ])
      if len(unique_concrete_signatures) != 1:
        raise ValueError(
            "Expected exactly one unique concrete_function signature, found "
            f"the following: {all_concrete_signatures}")
      concrete_func = all_concrete_funcs[0]
    else:
      raise ValueError(f"No concrete functions found on {tf_func}")

    self._maybe_init_tables(concrete_func, name)

    def func(*args, **kwargs):
      bound_args = self.signatures[name].bind(*args, **kwargs)
      canonical_args = concrete_func.structured_input_signature

      flat_bound_args = tree.flatten_with_path(
          (bound_args.args, bound_args.kwargs))
      flat_canonical_args = tree.flatten_with_path(canonical_args)

      # Check for missing arguments.
      flat_bound_args_dict = dict(flat_bound_args)
      for arg_path, arg_spec in flat_canonical_args:
        if arg_path in flat_bound_args_dict and arg_spec is None:
          arg_value = flat_bound_args_dict[arg_path]
          if arg_value is not None:
            logging.log_first_n(
                logging.WARNING,
                "Received unexpected argument `%s` for path %s, replaced with "
                "None.",
                20,
                arg_value, arg_path)
          flat_bound_args_dict[arg_path] = None

      # Filter out extraneous arguments and dictionary keys.
      flat_canonical_args_dict = dict(flat_canonical_args)
      filtered_flat_bound_args = {
          arg_path: arg_value
          for arg_path, arg_value in flat_bound_args_dict.items()
          if arg_path in flat_canonical_args_dict
      }
      full_flat_bound_args = [
          filtered_flat_bound_args.get(arg_path, None)
          for arg_path, _ in flat_canonical_args
      ]
      filtered_args, filtered_kwargs = tree.unflatten_as(
          canonical_args, full_flat_bound_args)

      return tf_func(*filtered_args, **filtered_kwargs)

    return _Function(
        func,
        copy.deepcopy(self.signatures[name]),
        copy.deepcopy(concrete_func.structured_input_signature),
    )
