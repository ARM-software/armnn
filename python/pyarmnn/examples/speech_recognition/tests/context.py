# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'common'))
import utils as common_utils

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import audio_capture
import audio_utils
import preprocess
