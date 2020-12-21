import os
from pathlib import Path

from setuptools import setup

# Todo finish and debug setup
models_path = Path('data/amcp_models')
prediction_path = Path('data/amcp_predictions')
if not os.path.exists(models_path):
    os.makedirs(models_path)
if not os.path.exists(models_path):
    os.mkdir(prediction_path)

setup(
    setup_requires=['setup.cfg'],
    setup_cfg=True
)