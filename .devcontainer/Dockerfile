# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.236.0/containers/python-3/.devcontainer/base.Dockerfile

ARG VARIANT="3.9"
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT}

# Install pytest
RUN pip install pytest

# Install lab2d
RUN pip install https://github.com/deepmind/lab2d/releases/download/release_candidate_2021-07-13/dmlab2d-1.0-cp39-cp39-manylinux_2_31_x86_64.whl

# Download assets
ADD https://storage.googleapis.com/dm-meltingpot/meltingpot-assets-1.0.0.tar.gz /workspaces/meltingpot/meltingpot/

# Set Python path
ENV PYTHONPATH="/workspaces/meltingpot"
