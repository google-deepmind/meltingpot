#!/bin/bash

python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -e .[dev]
pip install -r examples/requirements.txt
pytest examples

