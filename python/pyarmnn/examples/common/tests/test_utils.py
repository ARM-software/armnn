# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

import os

from context import cv_utils
from context import utils


def test_get_source_encoding(test_data_folder):
    video_file = os.path.join(test_data_folder, "Megamind.avi")
    video, video_writer, frame_count = cv_utils.init_video_file_capture(video_file, "/tmp")
    assert cv_utils.get_source_encoding_int(video) == 1145656920


def test_read_existing_labels_file(test_data_folder):
    label_file = os.path.join(test_data_folder, "labelmap.txt")
    labels_map = utils.dict_labels(label_file)
    assert labels_map is not None
