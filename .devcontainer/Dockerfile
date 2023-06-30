# Use Nvidia Ubuntu 20 base (includes CUDA if a supported GPU is present)
# https://hub.docker.com/r/nvidia/cuda
FROM nvidia/cuda:12.0.1-cudnn8-devel-ubuntu20.04@sha256:b10f2515f33613fe9eaacd03842d38093fedeb98b25170ae82a5ef77fe2a7e76

# Install dependencies
RUN apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get -qq -y install \
  build-essential \
  curl \
  ffmpeg \
  git \
  python3.9 \
  python3.9-dev \
  python3.9-distutils \
  rsync

# Install pip (we need the latest version not the standard Ubuntu version, to
# support modern wheels)
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.9 get-pip.py

# Set python aliases
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# Install python dev dependencies
RUN pip install \
  jedi \
  pytest \
  pylance \
  toml \
  yapf

# Install lab2d from wheel.
RUN pip install https://github.com/deepmind/lab2d/releases/download/release_candidate_2023-06_01/dmlab2d-1.0-cp39-cp39-manylinux_2_35_x86_64.whl

# Set Python path
ENV PYTHONPATH="/workspaces/meltingpot"
