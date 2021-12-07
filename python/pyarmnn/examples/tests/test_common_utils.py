# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

import os
import time

import cv2
import numpy as np
from context import cv_utils
from context import utils
from utils import Profiling


def test_get_source_encoding(test_data_folder):
    video_file = os.path.join(test_data_folder, "Megamind.avi")
    video, video_writer, frame_count = cv_utils.init_video_file_capture(video_file, "/tmp")
    assert cv_utils.get_source_encoding_int(video) == 1145656920


def test_read_existing_labels_file(test_data_folder):
    label_file = os.path.join(test_data_folder, "labelmap.txt")
    labels_map = utils.dict_labels(label_file)
    assert labels_map is not None


def test_preprocess(test_data_folder):
    content_image = "messi5.jpg"
    target_shape = (1, 256, 256, 3)
    padding = True
    image = cv2.imread(os.path.join(test_data_folder, content_image))
    image = cv_utils.preprocess(image, np.float32, target_shape, True, padding)

    assert image.shape == target_shape


def test_profiling():
    profiler = Profiling(True)
    profiler.profiling_start()
    time.sleep(1)
    period = profiler.profiling_stop_and_print_us("Sleep for 1 second")
    assert (1_000_000 < period < 1_002_000)

