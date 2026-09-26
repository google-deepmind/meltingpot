# Copyright 2026 DeepMind Technologies Limited.
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
"""Tests for permissive SavedModel wrapping."""

import types
from unittest import mock

from absl.testing import absltest
from meltingpot.utils.policies import permissive_model


class PermissiveModelTableTest(absltest.TestCase):

  def test_initializes_all_tables_before_marking_function_initialized(self):
    model = object.__new__(permissive_model.PermissiveModel)
    keys_a = mock.Mock(dtype=mock.sentinel.key_dtype_a)
    values_a = mock.Mock(dtype=mock.sentinel.value_dtype_a)
    keys_b = mock.Mock(dtype=mock.sentinel.key_dtype_b)
    values_b = mock.Mock(dtype=mock.sentinel.value_dtype_b)
    tables = {
        'table_a': (keys_a, values_a),
        'table_b': (keys_b, values_b),
    }
    model._tables = {'step': tables}
    model._initialized_tables = {}

    nodes = [
        types.SimpleNamespace(
            name='table_a',
            attr={'shared_name': types.SimpleNamespace(s=b'table-a')},
        ),
        types.SimpleNamespace(
            name='table_b',
            attr={'shared_name': types.SimpleNamespace(s=b'table-b')},
        ),
    ]
    graph_def = types.SimpleNamespace(
        node=nodes,
        library=types.SimpleNamespace(function=[]),
    )
    concrete_func = mock.Mock()
    concrete_func.graph.as_graph_def.return_value = graph_def

    with mock.patch.object(
        permissive_model.tf.raw_ops,
        'HashTableV2',
        side_effect=[mock.sentinel.handle_a, mock.sentinel.handle_b],
    ) as hash_table, mock.patch.object(
        permissive_model.tf.raw_ops, 'LookupTableImportV2'
    ) as import_table:
      model._maybe_init_tables(concrete_func, 'step')

    self.assertEqual(hash_table.call_count, 2)
    self.assertEqual(import_table.call_count, 2)
    self.assertNotIn('step', model._tables)
    self.assertIs(model._initialized_tables['step'], tables)


if __name__ == '__main__':
  absltest.main()
