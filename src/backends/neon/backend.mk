#
# Copyright © 2017 ARM Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

# BACKEND_SOURCES contains the list of files to be included
# in the Android build and it is picked up by the Android.mk
# file in the root of ArmNN

BACKEND_SOURCES := \
        NeonBackend.cpp \
        NeonInterceptorScheduler.cpp \
        NeonLayerSupport.cpp \
        NeonTimer.cpp \
        NeonWorkloadFactory.cpp \
        workloads/NeonActivationWorkload.cpp \
        workloads/NeonAdditionFloatWorkload.cpp \
        workloads/NeonBatchNormalizationFloatWorkload.cpp \
        workloads/NeonConstantWorkload.cpp \
        workloads/NeonConvertFp16ToFp32Workload.cpp \
        workloads/NeonConvertFp32ToFp16Workload.cpp \
        workloads/NeonConvolution2dWorkload.cpp \
        workloads/NeonDepthwiseConvolutionWorkload.cpp \
        workloads/NeonFloorFloatWorkload.cpp \
        workloads/NeonFullyConnectedWorkload.cpp \
        workloads/NeonL2NormalizationFloatWorkload.cpp \
        workloads/NeonLstmFloatWorkload.cpp \
        workloads/NeonMultiplicationFloatWorkload.cpp \
        workloads/NeonNormalizationFloatWorkload.cpp \
        workloads/NeonPermuteWorkload.cpp \
        workloads/NeonPooling2dWorkload.cpp \
        workloads/NeonReshapeWorkload.cpp \
        workloads/NeonSoftmaxBaseWorkload.cpp \
        workloads/NeonSoftmaxFloatWorkload.cpp \
        workloads/NeonSoftmaxUint8Workload.cpp \
        workloads/NeonSubtractionFloatWorkload.cpp

# BACKEND_TEST_SOURCES contains the list of files to be included
# in the Android unit test build (armnn-tests) and it is picked
# up by the Android.mk file in the root of ArmNN

BACKEND_TEST_SOURCES := \
        test/NeonCreateWorkloadTests.cpp \
        test/NeonEndToEndTests.cpp \
        test/NeonJsonPrinterTests.cpp \
        test/NeonLayerSupportTests.cpp \
        test/NeonLayerTests.cpp \
        test/NeonMemCopyTests.cpp \
        test/NeonOptimizedNetworkTests.cpp \
        test/NeonRuntimeTests.cpp \
        test/NeonTimerTest.cpp

