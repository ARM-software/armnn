#
# Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

# BACKEND_SOURCES contains the list of files to be included
# in the Android build and it is picked up by the Android.mk
# file in the root of ArmNN

# The variable to enable/disable the TOSA Reference backend
# (ARMNN_TOSA_REF_ENABLED is declared in android-nn-driver/Android.mk)
ifeq ($(ARMNN_TOSA_REF_ENABLED),1)

# ARMNN_TOSA_REF_ENABLED == 1
# Include the source files for the TOSA reference backend

BACKEND_SOURCES := \
        TosaRefBackend.cpp \
        TosaRefLayerSupport.cpp \
        TosaRefMemoryManager.cpp \
        TosaRefRegistryInitializer.cpp \
        TosaRefTensorHandle.cpp \
        TosaRefTensorHandleFactory.cpp \
        TosaRefWorkloadFactory.cpp \
        workloads/TosaRefPreCompiledWorkload.cpp
else

# ARMNN_TOSA_REF_ENABLED == 0
# No source file will be compiled for the reference backend

BACKEND_SOURCES :=

endif

# BACKEND_TEST_SOURCES contains the list of files to be included
# in the Android unit test build (armnn-tests) and it is picked
# up by the Android.mk file in the root of ArmNN

# The variable to enable/disable the TOSA Reference backend
# (ARMNN_TOSA_REF_ENABLED is declared in android-nn-driver/Android.mk)
ifeq ($(ARMNN_TOSA_REF_ENABLED),1)

# ARMNN_TOSA_REF_ENABLED == 1
# Include the source files for the TOSA Reference backend tests

BACKEND_TEST_SOURCES := \
        test/TosaRefLayerSupportTests.cpp \
        test/TosaRefLayerTests.cpp
else

# ARMNN_TOSA_REF_ENABLED == 0
# No source file will be compiled for the TOSA reference backend tests

BACKEND_TEST_SOURCES :=

endif
