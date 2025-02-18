# -*- coding: utf-8 -*-

import os
import codecs
from setuptools import find_packages, setup


readme_file = "README.md"
version_file = "tdrec/version.py"


def get_readme():
    """
    Parse readme content.
    """
    with codecs.open(readme_file, encoding="utf-8") as f:
        content = f.read()
    return content


def get_version():
    """
    Get TorchDeepRec version.
    """
    with codecs.open(version_file, encoding="utf-8") as f:
        exec(compile(f.read(), version_file, "exec"))
    if "NIGHTLY_VERSION" in os.environ:
        return f'{locals()["__version__"]}+{os.environ["NIGHTLY_VERSION"]}'
    else:
        return locals()["__version__"]


setup(
    name="tdrec",
    version=get_version(),
    description="A deep learning recommendation algorithm framework based on PyTorch.",
    long_description=get_readme(),
    packages=find_packages(),
    python_requires=">=3.8",
    include_package_data=True,
    classifiers=["Programming Language :: Python :: 3"],
)
