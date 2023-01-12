#
# Copyright © 2017,2023 Arm Ltd and Contributors.  All rights reserved.
# SPDX-License-Identifier: MIT
#

# COMMON_SOURCES contains the list of files to be included
# in the Android build and it is picked up by the Android.mk
# file in the root of ArmNN

COMMON_SOURCES := \
    ArmComputeTensorUtils.cpp \
    BaseMemoryManager.cpp

ifeq ($(ARMNN_COMPUTE_CL_ENABLED),1)
COMMON_SOURCES += \
    ArmComputeTuningUtils.cpp
endif # ARMNN_COMPUTE_CL_ENABLED == 1

# COMMON_TEST_SOURCES contains the list of files to be included
# in the Android unit test build (armnn-tests) and it is picked
# up by the Android.mk file in the root of ArmNN

COMMON_TEST_SOURCES := \
    test/ArmComputeTensorUtilsTests.cpp \
    test/MemCopyTests.cpp
