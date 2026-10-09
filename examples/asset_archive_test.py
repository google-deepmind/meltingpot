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
"""Offline validation of setup.py's streamed Melting Pot asset extraction."""

import contextlib
import io
import os
from pathlib import Path
import runpy
import stat
import tarfile
import tempfile
import unittest
from unittest import mock

import setuptools

ROOT = Path(__file__).resolve().parents[1]


class AssetArchiveExtractionTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    # Read the real build command without running setuptools.setup/install.
    previous_cwd = os.getcwd()
    os.chdir(ROOT)
    try:
      with mock.patch.object(setuptools, 'setup'):
        cls.build_type = runpy.run_path(
            str(ROOT / 'setup.py'), run_name='asset_archive_test'
        )['BuildPy']
    finally:
      os.chdir(previous_cwd)

  def setUp(self):
    super().setUp()
    self.tempdir = tempfile.TemporaryDirectory()
    self.addCleanup(self.tempdir.cleanup)
    self.root = Path(self.tempdir.name)
    self.assets_dir = self.root / 'meltingpot' / 'assets'
    self.assets_dir.mkdir(parents=True)
    (self.assets_dir / 'old.txt').write_bytes(b'existing assets')
    self.tar_path = self.root / 'assets.tar.gz'
    self.builder = self.build_type(setuptools.Distribution())
    self.builder.get_package_dir = lambda _name: str(self.root)

  def _create_tarball(self, entries):
    with tarfile.open(self.tar_path, mode='w:gz') as archive:
      for name, value in entries:
        info = tarfile.TarInfo(name)
        if isinstance(value, bytes):
          info.size = len(value)
          info.mode = 0o7777
          archive.addfile(info, io.BytesIO(value))
        elif value == 'directory':
          info.type = tarfile.DIRTYPE
          archive.addfile(info)
        elif value == 'symlink':
          info.type = tarfile.SYMTYPE
          info.linkname = '../../outside.txt'
          archive.addfile(info)
        elif value == 'hardlink':
          info.type = tarfile.LNKTYPE
          info.linkname = '../../outside.txt'
          archive.addfile(info)
        elif value == 'fifo':
          info.type = tarfile.FIFOTYPE
          archive.addfile(info)
        else:
          raise AssertionError(f'Unknown archive type {value!r}')

  def _extract(self):
    with contextlib.redirect_stdout(io.StringIO()):
      self.builder.extract_assets(str(self.tar_path))

  def _assert_original_unchanged(self):
    self.assertEqual(
        (self.assets_dir / 'old.txt').read_bytes(), b'existing assets'
    )
    self.assertEqual(
        list((self.root / 'meltingpot').glob('.assets-staging-*')), []
    )

  def test_valid_archive_installs_nested_assets_and_replaces_old(self):
    self._create_tarball([
        ('assets/', 'directory'),
        ('assets/saved_models/', 'directory'),
        ('assets/saved_models/model.pb', b'valid model'),
    ])
    self._extract()
    self.assertEqual(
        (self.assets_dir / 'saved_models' / 'model.pb').read_bytes(),
        b'valid model',
    )
    self.assertFalse((self.assets_dir / 'old.txt').exists())
    self.assertEqual(
        list((self.root / 'meltingpot').glob('.assets-staging-*')), []
    )
    mode = (self.assets_dir / 'saved_models' / 'model.pb').stat().st_mode
    self.assertFalse(mode & stat.S_ISUID)

  def test_empty_destination_is_installed(self):
    self.assets_dir.rename(self.root / 'meltingpot' / 'old-install')
    self._create_tarball([('assets/new.txt', b'new install')])
    self._extract()
    self.assertEqual((self.assets_dir / 'new.txt').read_bytes(), b'new install')

  def test_parent_directory_traversal_is_rejected(self):
    self._create_tarball([
        ('assets/safe.txt', b'safe'),
        ('assets/../../outside.txt', b'injected'),
    ])
    with self.assertRaisesRegex(ValueError, 'Unsafe.*outside.txt'):
      self._extract()
    self._assert_original_unchanged()
    self.assertFalse((self.root / 'outside.txt').exists())

  def test_absolute_path_is_rejected(self):
    self._create_tarball([
        ('assets/safe.txt', b'safe'),
        ('/tmp/meltingpot-unexpected.txt', b'unexpected'),
    ])
    with self.assertRaisesRegex(ValueError, 'Unsafe'):
      self._extract()
    self._assert_original_unchanged()

  def test_wrong_archive_root_is_rejected(self):
    self._create_tarball([('other/asset.txt', b'not assets')])
    with self.assertRaisesRegex(ValueError, 'Unsafe'):
      self._extract()
    self._assert_original_unchanged()

  def test_windows_style_escape_is_rejected(self):
    self._create_tarball([('assets/..\\..\\windows.txt', b'unsafe')])
    with self.assertRaisesRegex(ValueError, 'Unsafe'):
      self._extract()
    self._assert_original_unchanged()

  def test_symlink_cannot_escape_staging(self):
    self._create_tarball([
        ('assets/safe.txt', b'good'),
        ('assets/link', 'symlink'),
    ])
    with self.assertRaisesRegex(ValueError, 'Unsafe'):
      self._extract()
    self._assert_original_unchanged()

  def test_hardlink_is_rejected(self):
    self._create_tarball([('assets/link', 'hardlink')])
    with self.assertRaisesRegex(ValueError, 'Unsafe'):
      self._extract()
    self._assert_original_unchanged()

  def test_fifo_is_rejected(self):
    self._create_tarball([('assets/pipe', 'fifo')])
    with self.assertRaisesRegex(ValueError, 'Unsafe'):
      self._extract()
    self._assert_original_unchanged()

  def test_archive_with_no_files_does_not_remove_existing_assets(self):
    self._create_tarball([('assets/', 'directory')])
    with self.assertRaisesRegex(ValueError, 'no asset files'):
      self._extract()
    self._assert_original_unchanged()

  def test_corrupt_archive_does_not_remove_existing_assets(self):
    self.tar_path.write_bytes(b'not a tarball')
    with self.assertRaises(tarfile.TarError):
      self._extract()
    self._assert_original_unchanged()

  def test_failed_install_restores_existing_assets(self):
    self._create_tarball([('assets/new.txt', b'new')])
    original_replace = os.replace

    def replace_with_failure(source, target):
      if Path(source).name == 'assets' and Path(target) == self.assets_dir:
        raise OSError('simulated replace failure')
      return original_replace(source, target)

    with mock.patch('os.replace', side_effect=replace_with_failure):
      with self.assertRaisesRegex(OSError, 'simulated replace failure'):
        self._extract()
    self._assert_original_unchanged()


if __name__ == '__main__':
  unittest.main()
