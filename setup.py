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

import setuptools

setuptools.setup(
    name='dm-meltingpot',
    version='1.0.2',
    license='Apache 2.0',
    license_files=['LICENSE'],
    url='https://github.com/deepmind/meltingpot',
    download_url='https://github.com/deepmind/meltingpot',
    author='DeepMind',
    author_email='noreply@google.com',
    description='A suite of test scenarios for multi-agent reinforcement learning.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
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
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],

    packages=['meltingpot'],
    package_data={
        'meltingpot': [
            'assets/saved_models/**/**/saved_model.pb',
            'assets/saved_models/**/**/variables/variables.data-00000-of-00001',
            'assets/saved_models/**/**/variables/variables.index',
            'lua/modules/*',
            'lua/levels/**/*',
        ],
    },
    python_requires='>=3.7',
    install_requires=[
        'absl-py',
        'chex',
        'dm_env',
        # 'dmlab2d',  # Not yet available for PIP install.
        'immutabledict',
        'ml-collections',
        'networkx',
        'numpy',
        'pygame',
        'rx',
        'tensorflow',
    ],
    extras_require={
        'examples': [
            'gym',
            'ray[rllib,default]',
        ],
    },
)
