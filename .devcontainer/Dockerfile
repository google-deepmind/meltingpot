# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.236.0/containers/python-3/.devcontainer/base.Dockerfile

# [Choice] Python version (use -bullseye variants on local arm64/Apple Silicon): 3, 3.10, 3.9, 3.8, 3.7, 3.6, 3-bullseye, 3.10-bullseye, 3.9-bullseye, 3.8-bullseye, 3.7-bullseye, 3.6-bullseye, 3-buster, 3.10-buster, 3.9-buster, 3.8-buster, 3.7-buster, 3.6-buster
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
