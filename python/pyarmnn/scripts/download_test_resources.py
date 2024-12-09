#!/usr/bin/env python3
# Copyright 2020 NXP
# SPDX-License-Identifier: MIT
"""Downloads and extracts resources for unit tests.

It is mandatory to run this script prior to running unit tests. Resources are stored as a tar.gz or a tar.bz2 archive and
extracted into the test/testdata/shared folder.
"""

import tarfile
import requests
import os
import uuid

SCRIPTS_DIR = os.path.dirname(os.path.realpath(__file__))
EXTRACT_DIR = os.path.join(SCRIPTS_DIR, "..", "test")
ARCHIVE_URL = "https://snapshots.linaro.org/components/pyarmnn-tests/pyarmnn_testdata_201100_20201022.tar.bz2"


def download_resources(url, save_path):
    # download archive - only support tar.gz or tar.bz2
    print("Downloading '{}'".format(url))
    temp_filename = str(uuid.uuid4())
    if url.endswith(".tar.bz2"):
        temp_filename += ".tar.bz2"
    elif url.endswith(".tar.gz"):
        temp_filename += ".tar.gz"
    else:
        raise RuntimeError("Unsupported file.")
    try:
        r = requests.get(url, stream=True)
    except requests.exceptions.RequestException as e:
        raise RuntimeError("Unable to download file: {}".format(e))
    file_path = os.path.join(save_path, temp_filename)
    with open(file_path, 'wb') as f:
        f.write(r.content)

    # extract and delete temp file
    with tarfile.open(file_path, "r:bz2" if temp_filename.endswith(".tar.bz2") else "r:gz") as tar:
        print("Extracting '{}'".format(file_path))
        tar.extractall(save_path)
    if os.path.exists(file_path):
        print("Removing '{}'".format(file_path))
        os.remove(file_path)


download_resources(ARCHIVE_URL, EXTRACT_DIR)
