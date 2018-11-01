#
# Copyright © 2017 ARM Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

# COMMON_SOURCES contains the list of files to be included
# in the Android build and it is picked up by the Android.mk
# file in the root of ArmNN

COMMON_SOURCES := \
    BackendContextRegistry.cpp \
    BackendRegistry.cpp \
    CpuTensorHandle.cpp \
    ILayerSupport.cpp \
    MemCopyWorkload.cpp \
    LayerSupportRegistry.cpp \
    OutputHandler.cpp \
    StringMapping.cpp \
    WorkloadData.cpp \
    WorkloadFactory.cpp

# COMMON_TEST_SOURCES contains the list of files to be included
# in the Android unit test build (armnn-tests) and it is picked
# up by the Android.mk file in the root of ArmNN

COMMON_TEST_SOURCES := \
    test/WorkloadDataValidation.cpp \
    test/TensorCopyUtils.cpp \
    test/LayerTests.cpp

