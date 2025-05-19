#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
from setuptools import setup, find_packages

# Read the version from __init__.py
with open(os.path.join('agent_suite', '__init__.py'), 'r') as f:
    version_file = f.read()
    version_match = re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', version_file, re.M)
    if version_match:
        version = version_match.group(1)
    else:
        version = '0.1.0'

# Read requirements from requirements.txt
with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(
    name="agent_suite",
    version=version,
    description="A comprehensive toolkit for building AI agents",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Agent Suite Contributors",
    author_email="example@example.com",
    url="https://github.com/example/agent_suite",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
) 