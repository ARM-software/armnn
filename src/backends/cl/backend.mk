#
# Copyright © 2017 ARM Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

# BACKEND_SOURCES contains the list of files to be included
# in the Android build and it is picked up by the Android.mk
# file in the root of ArmNN

BACKEND_SOURCES := \
        ClBackend.cpp \
        ClBackendContext.cpp \
        ClContextControl.cpp \
        ClLayerSupport.cpp \
        ClWorkloadFactory.cpp \
        OpenClTimer.cpp \
        workloads/ClActivationWorkload.cpp \
        workloads/ClAdditionWorkload.cpp \
        workloads/ClBatchNormalizationFloatWorkload.cpp \
        workloads/ClBatchToSpaceNdWorkload.cpp \
        workloads/ClConstantWorkload.cpp \
        workloads/ClConvertFp16ToFp32Workload.cpp \
        workloads/ClConvertFp32ToFp16Workload.cpp \
        workloads/ClConvolution2dWorkload.cpp \
        workloads/ClDepthwiseConvolutionWorkload.cpp \
        workloads/ClDivisionFloatWorkload.cpp \
        workloads/ClFloorFloatWorkload.cpp \
        workloads/ClFullyConnectedWorkload.cpp \
        workloads/ClL2NormalizationFloatWorkload.cpp \
        workloads/ClLstmFloatWorkload.cpp \
        workloads/ClMaximumWorkload.cpp \
        workloads/ClMeanWorkload.cpp \
        workloads/ClMergerWorkload.cpp \
        workloads/ClMultiplicationWorkload.cpp \
        workloads/ClNormalizationFloatWorkload.cpp \
        workloads/ClPadWorkload.cpp \
        workloads/ClPermuteWorkload.cpp \
        workloads/ClPooling2dWorkload.cpp \
        workloads/ClReshapeWorkload.cpp \
        workloads/ClResizeBilinearFloatWorkload.cpp \
        workloads/ClSoftmaxBaseWorkload.cpp \
        workloads/ClSoftmaxFloatWorkload.cpp \
        workloads/ClSoftmaxUint8Workload.cpp \
        workloads/ClStridedSliceWorkload.cpp \
        workloads/ClSubtractionWorkload.cpp

# BACKEND_TEST_SOURCES contains the list of files to be included
# in the Android unit test build (armnn-tests) and it is picked
# up by the Android.mk file in the root of ArmNN

BACKEND_TEST_SOURCES := \
        test/ClCreateWorkloadTests.cpp \
        test/ClEndToEndTests.cpp \
        test/ClJsonPrinterTests.cpp \
        test/ClLayerSupportTests.cpp \
        test/ClLayerTests.cpp \
        test/ClMemCopyTests.cpp \
        test/ClOptimizedNetworkTests.cpp \
        test/ClRuntimeTests.cpp \
        test/Fp16SupportTest.cpp \
        test/OpenClTimerTest.cpp
