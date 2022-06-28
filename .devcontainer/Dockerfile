# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.236.0/containers/python-3/.devcontainer/base.Dockerfile
# Note bullseye version is multi-architecture (supports arm64)
ARG VARIANT="3.9-bullseye"
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT}

# Install pytest
RUN pip install pytest

# Install lab2d (appropriate version for architecture)
RUN if [ "$(uname -m)" != 'x86_64' ]; then \
    echo "No Lab2d wheel available for $(uname -m) machines." >&2 \
    exit 1; \
  elif [ "$(uname -s)" = 'Linux' ]; then \
    pip install https://github.com/deepmind/lab2d/releases/download/release_candidate_2022-03-24/dmlab2d-1.0-cp39-cp39-manylinux_2_31_x86_64.whl ;\
  else \
    pip install https://github.com/deepmind/lab2d/releases/download/release_candidate_2022-03-24/dmlab2d-1.0-cp39-cp39-macosx_10_15_x86_64.whl ;\
  fi

# Download assets
ADD https://storage.googleapis.com/dm-meltingpot/meltingpot-assets-1.0.0.tar.gz /workspaces/meltingpot/meltingpot/

# Set Python path
ENV PYTHONPATH="/workspaces/meltingpot"
