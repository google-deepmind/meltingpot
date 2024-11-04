# Copyright 2024 DeepMind Technologies Limited.
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
"""Subject that emits a video at the end of each episode."""

import os
import uuid

import cv2
import dm_env
import numpy as np
from reactivex import subject


class VideoSubject(subject.Subject):
  """Subject that emits a video at the end of each episode."""

  def __init__(
      self,
      root: str,
      *,
      extension: str = 'webm',
      codec: str = 'vp90',
      fps: int = 30,
  ) -> None:
    """Initializes the instance.

    Args:
      root: directory to write videos in.
      extension: file extention of file.
      codec: codex to write with.
      fps: frames-per-second for videos.

    Raises:
      FileNotFoundError: if the root directory does not exist.
    """
    super().__init__()
    self._root = root
    if not os.path.exists(root):
      raise FileNotFoundError(f'Video root {root!r} does not exist.')
    self._extension = extension
    self._codec = codec
    self._fps = fps
    self._path = None
    self._writer = None

  def on_next(self, timestep: dm_env.TimeStep) -> None:
    """Called on each timestep.

    Args:
      timestep: the most recent timestep.
    """
    rgb_frame = timestep.observation[0]['WORLD.RGB']
    height, width, colors = rgb_frame.shape
    if colors != 3:
      raise ValueError('WORLD.RGB is not RGB.')
    if rgb_frame.dtype != np.uint8:
      raise ValueError('WORLD.RGB is not uint8.')
    if rgb_frame.min() < 0 or rgb_frame.max() > 255:
      raise ValueError('WORLD.RGB is not in [0, 255].')

    if timestep.step_type.first():
      self._path = os.path.join(
          self._root, f'{uuid.uuid4().hex}.{self._extension}')
      self._writer = cv2.VideoWriter(
          filename=self._path,
          fourcc=cv2.VideoWriter_fourcc(*self._codec),
          fps=self._fps,
          frameSize=(width, height),
          isColor=True)
    elif self._writer is None:
      raise ValueError('First timestep must be StepType.FIRST.')
    bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    assert self._writer.isOpened()  # Catches any cv2 usage errors.
    self._writer.write(bgr_frame)
    if timestep.step_type.last():
      self._writer.release()
      super().on_next(self._path)
      self._path = None
      self._writer = None

  def dispose(self):
    """See base class."""
    if self._writer is not None:
      self._writer.release()
    super().dispose()
