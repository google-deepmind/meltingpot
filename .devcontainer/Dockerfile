# Use Nvidia Ubuntu 22 base (includes CUDA if a supported GPU is present)
# https://hub.docker.com/r/nvidia/cuda
FROM nvidia/cuda:12.6.1-devel-ubuntu22.04@sha256:7a205faa1359c3eeccc283a5839f75166404e3e4b42d6a48ebe37ade8cf76ade

ARG PYTHON_VERSION

# Install dependencies
RUN apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get -qq -y install \
  build-essential \
  curl \
  ffmpeg \
  git \
  python${PYTHON_VERSION} \
  python${PYTHON_VERSION}-dev \
  python${PYTHON_VERSION}-distutils \
  rsync

# Set python aliases
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1

# Install pip (we need the latest version not the standard Ubuntu version, to
# support modern wheels)
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python get-pip.py

# Install setuptools.
RUN pip install setuptools
