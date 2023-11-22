#
# Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

# BACKEND_SOURCES contains the list of files to be included
# in the Android build and it is picked up by the Android.mk
# file in the root of ArmNN

# The variable to enable/disable the GPU Dynamic Fusion backend
# (ARMNN_COMPUTE_GPUFSA_ENABLED is declared in android-nn-driver/Android.mk)
ifeq ($(ARMNN_COMPUTE_GPUFSA_ENABLED),1)

# ARMNN_COMPUTE_GPUFSA_ENABLED == 1
# Include the source files for the GPU Dynamic Fusion backend

BACKEND_SOURCES := \
        GpuFsaBackend.cpp \
        GpuFsaBackendContext.cpp \
        GpuFsaContextControl.cpp \
        GpuFsaLayerSupport.cpp \
        GpuFsaRegistryInitializer.cpp \
        GpuFsaTensorHandleFactory.cpp \
        GpuFsaWorkloadFactory.cpp \
        layerValidators/GpuFsaConvolution2dValidate.cpp
else

# ARMNN_COMPUTE_GPUFSA_ENABLED == 0
# No source file will be compiled for the GPU Dynamic Fusion backend

BACKEND_SOURCES :=

endif

# BACKEND_TEST_SOURCES contains the list of files to be included
# in the Android unit test build (armnn-tests) and it is picked
# up by the Android.mk file in the root of ArmNN

# The variable to enable/disable the GPU Dynamic Fusion backend
# (ARMNN_COMPUTE_GPUFSA_ENABLED is declared in android-nn-driver/Android.mk)
ifeq ($(ARMNN_COMPUTE_GPUFSA_ENABLED),1)

# ARMNN_COMPUTE_GPUFSA_ENABLED == 1
# Include the source files for the GPU Dynamic Fusion backend tests

BACKEND_TEST_SOURCES := \
		test/GpuFsaEndToEndTests.cpp \
        test/GpuFsaLayerSupportTests.cpp \
        test/GpuFsaLayerTests.cpp \
        test/GpuFsaOptimizedNetworkTests.cpp
else

# ARMNN_COMPUTE_GPUFSA_ENABLED == 0
# No source file will be compiled for the GPU Dynamic Fusion backend tests

BACKEND_TEST_SOURCES :=

endif
