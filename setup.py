import os
from setuptools import find_packages, setup


__version__ = "0.1"

if "VERSION" in os.environ:
    BUILD_NUMBER = os.environ["VERSION"].rsplit(".", 1)[-1]
else:
    BUILD_NUMBER = os.environ.get("BUILD_NUMBER", "dev")

dependencies = [
    "click",
    "numpy",
    "pandas",
    "scikit-learn",
    "lightgbm",
    "black",
    "flask",
    "flask-caching",
    "requests",
    "optuna",
]

setup(
    name="demand_sense",
    version="{0}.{1}".format(__version__, BUILD_NUMBER),
    description="Demand Sensing",
    author="Deepan Chakravarthi Padmanabhan",
    install_requires=dependencies,
    packages=find_packages(),
    zip_safe=False,
    entry_points=dict(
        console_scripts=[
            "demand_sense_train=demand_sense.trainer:train",
        ]
    ),
    python_requires=">=3.7",
)
