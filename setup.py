# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os

from setuptools import find_packages, setup

# Declare your non-python data files:
# Files underneath configuration/ will be copied into the build preserving the
# subdirectory structure if they exist.
data_files = []
for root, dirs, files in os.walk("configuration"):
    data_files.append(
        (os.path.relpath(root, "configuration"), [os.path.join(root, f) for f in files])
    )

setup(
    name="CompMedDsumEval",
    version="1.0",
    # declare your packages
    packages=find_packages(where="src", exclude=("test",)),
    package_dir={"": "src"},
    # include data files
    data_files=data_files,
    # defines files which should be bundled with the python code for redistribution
    package_data={"": ["py.typed"]},
    root_script_source_version="default-only",
    check_format=False,  # Enable build-time format checking
    test_mypy=False,  # Enable type checking
    test_flake8=False,  # Enable linting at build time
)
