# Use Nvidia Ubuntu 22 base (includes CUDA if a supported GPU is present)
# https://hub.docker.com/r/nvidia/cuda
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04@sha256:7e81c70fb55bfc54e6711700eb721bb94159a3cd546620b37210d12c17914fa6

# Install dependencies
RUN apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get -qq -y install \
  build-essential \
  curl \
  ffmpeg \
  git \
  python3.10 \
  python3.10-dev \
  python3.10-distutils \
  rsync

# Set python aliases
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install pip (we need the latest version not the standard Ubuntu version, to
# support modern wheels)
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python get-pip.py

# Install python dev dependencies
RUN pip install \
  jedi \
  pytest \
  pylance \
  toml \
  yapf

# Install lab2d from wheel.
RUN pip install https://github.com/deepmind/lab2d/releases/download/release_candidate_2023-06_01/dmlab2d-1.0-cp310-cp310-manylinux_2_35_x86_64.whl

# Set Python path
ENV PYTHONPATH="/workspaces/meltingpot"
