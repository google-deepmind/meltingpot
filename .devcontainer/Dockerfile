# Use Nvidia Ubuntu 22 base (includes CUDA if a supported GPU is present)
# https://hub.docker.com/r/nvidia/cuda
FROM nvidia/cuda:13.4.1-cudnn-devel-ubuntu24.04@sha256:4862208a2c62238d7b8dd26e525293d5fce4b35c4b3ae90407e75057d669346c

ARG PYTHON_VERSION

# Install dependencies
RUN apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common && \
  add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get -qq -y install \
  build-essential \
  curl \
  ffmpeg \
  git \
  python${PYTHON_VERSION} \
  python${PYTHON_VERSION}-dev \
  rsync

# Set python aliases
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1

# Install pip (we need the latest version not the standard Ubuntu version, to
# support modern wheels)
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python get-pip.py

# Install setuptools.
RUN pip install setuptools
