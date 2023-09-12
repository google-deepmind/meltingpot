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
"""Install script for setuptools."""

import os
import shutil
import tarfile
import urllib.request

import setuptools
from setuptools.command import build_py

VERSION = '2.2.0'
ASSETS_VERSION = '2.1.0'

ASSETS_URL = f'http://storage.googleapis.com/dm-meltingpot/meltingpot-assets-{ASSETS_VERSION}.tar.gz'


class BuildPy(build_py.build_py):
  """Command that downloads Melting Pot assets as part of build_py."""

  def run(self):
    self.download_and_extract_assets()
    if not self.editable_mode:
      super().run()
      self.build_assets()

  def download_and_extract_assets(self):
    """Downloads and extracts assets to meltingpot/assets."""
    tar_file_path = os.path.join(
        self.get_package_dir('assets'), os.path.basename(ASSETS_URL))
    if os.path.exists(tar_file_path):
      print(f'found cached assets {tar_file_path}', flush=True)
    else:
      os.makedirs(os.path.dirname(tar_file_path), exist_ok=True)
      print('downloading assets...', flush=True)
      urllib.request.urlretrieve(ASSETS_URL, filename=tar_file_path)
      print(f'downloaded {tar_file_path}', flush=True)

    root = os.path.join(self.get_package_dir(''), 'meltingpot')
    os.makedirs(root, exist_ok=True)
    if os.path.exists(f'{root}/assets'):
      shutil.rmtree(f'{root}/assets')
      print('deleted existing assets', flush=True)
    with tarfile.open(tar_file_path, mode='r|*') as tarball:
      tarball.extractall(root)
    print(f'extracted assets from {tar_file_path} to {root}/assets', flush=True)

  def build_assets(self):
    """Copies assets from package to build lib."""
    package_root = os.path.join(self.get_package_dir(''), 'meltingpot')
    os.makedirs(package_root, exist_ok=True)
    build_root = os.path.join(self.build_lib, 'meltingpot')
    if os.path.exists(f'{build_root}/assets'):
      shutil.rmtree(f'{build_root}/assets')
      print('deleted existing assets', flush=True)
    shutil.copytree(f'{package_root}/assets', f'{build_root}/assets')
    print(f'copied assets from {package_root}/assets to {build_root}/assets',
          flush=True)


setuptools.setup(
    name='dm-meltingpot',
    version=VERSION,
    license='Apache 2.0',
    license_files=['LICENSE'],
    url='https://github.com/google-deepmind/meltingpot',
    download_url='https://github.com/google-deepmind/meltingpot/releases',
    author='DeepMind',
    author_email='noreply@google.com',
    description=(
        'A suite of test scenarios for multi-agent reinforcement learning.'),
    keywords='multi-agent reinforcement-learning python machine-learning',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    cmdclass={'build_py': BuildPy},
    package_dir={
        'meltingpot': 'meltingpot',
    },
    package_data={
        'meltingpot.lua': ['**'],
    },
    python_requires='>=3.10',
    install_requires=[
        'absl-py',
        'chex<0.1.81',  # Incompatible with tensorflow 2.13 (due to numpy req).
        'dm-env',
        'dmlab2d',
        'dm-tree',
        'immutabledict',
        'ml-collections',
        'networkx',
        'numpy',
        'opencv-python',
        'pandas',
        'pygame',
        'reactivex',
        'tensorflow',
    ],
    extras_require={
        # Used in development.
        'dev': [
            'build',
            'isort',
            'pipreqs',
            'pyink',
            'pylint',
            'pytest-xdist',
            'pytype',
        ],
    },
)
