#!/usr/bin/env python3
# Copyright © 2020 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
"""Generate PyArmNN documentation."""

import os
import tarfile

import pyarmnn as ann
import shutil

from typing import List, Union

from pdoc.cli import main


def __copy_file_to_dir(file_paths: Union[List[str], str], target_dir_path: str):
    file_paths = [] + file_paths

    if not (os.path.exists(target_dir_path) and os.path.isdir(target_dir_path)):
        os.makedirs(target_dir_path)

    for file_path in file_paths:
        if not (os.path.exists(file_path) and os.path.isfile(file_path)):
            raise RuntimeError('Not a file: {}'.format(file_path))

        file_name = os.path.basename(file_path)
        shutil.copyfile(file_path, os.path.join(str(target_dir_path), file_name))


def copy_doc_images():
    __copy_file_to_dir(file_paths=['../../docs/pyarmnn.png'],
                       target_dir_path='docs')


def archive_docs(path, version):

    output_filename = f'pyarmnn_docs-{version}.tar'

    with tarfile.open(output_filename, "w") as tar:
        tar.add(path)


if __name__ == "__main__":
    with open('./README.md', 'r') as readme_file:
        top_level_pyarmnn_doc = ''.join(readme_file.readlines())
        ann.__doc__ = top_level_pyarmnn_doc

    main()

    copy_doc_images()
    archive_docs('./docs', ann.__version__)
