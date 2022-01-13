#
# Copyright © 2017 ARM Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

# BACKEND_SOURCES contains the list of files to be included
# in the Android build and it is picked up by the Android.mk
# file in the root of ArmNN

# The variable to enable/disable the NEON backend (ARMNN_COMPUTE_NEON_ENABLED is declared in android-nn-driver/Android.mk)
ifeq ($(ARMNN_COMPUTE_NEON_ENABLED),1)

# ARMNN_COMPUTE_NEON_ENABLED == 1
# Include the source files for the NEON backend

BACKEND_SOURCES := \
        NeonBackend.cpp \
        NeonBackendModelContext.cpp \
        NeonInterceptorScheduler.cpp \
        NeonLayerSupport.cpp \
        NeonRegistryInitializer.cpp \
        NeonTensorHandleFactory.cpp \
        NeonTimer.cpp \
        NeonWorkloadFactory.cpp \
        workloads/NeonAbsWorkload.cpp \
        workloads/NeonActivationWorkload.cpp \
        workloads/NeonAdditionWorkload.cpp \
        workloads/NeonArgMinMaxWorkload.cpp \
        workloads/NeonBatchNormalizationWorkload.cpp \
        workloads/NeonBatchToSpaceNdWorkload.cpp \
        workloads/NeonCastWorkload.cpp \
        workloads/NeonChannelShuffleWorkload.cpp \
        workloads/NeonComparisonWorkload.cpp \
        workloads/NeonConcatWorkload.cpp \
        workloads/NeonConstantWorkload.cpp \
        workloads/NeonConvertBf16ToFp32Workload.cpp \
        workloads/NeonConvertFp32ToBf16Workload.cpp \
        workloads/NeonConvertFp16ToFp32Workload.cpp \
        workloads/NeonConvertFp32ToFp16Workload.cpp \
        workloads/NeonConvolution2dWorkload.cpp \
        workloads/NeonConvolution3dWorkload.cpp \
        workloads/NeonDepthToSpaceWorkload.cpp \
        workloads/NeonDepthwiseConvolutionWorkload.cpp \
        workloads/NeonDequantizeWorkload.cpp \
        workloads/NeonDetectionPostProcessWorkload.cpp \
        workloads/NeonExpWorkload.cpp \
        workloads/NeonFillWorkload.cpp \
        workloads/NeonFloorFloatWorkload.cpp \
        workloads/NeonFullyConnectedWorkload.cpp \
        workloads/NeonGatherWorkload.cpp \
        workloads/NeonInstanceNormalizationWorkload.cpp \
        workloads/NeonL2NormalizationFloatWorkload.cpp \
        workloads/NeonLogWorkload.cpp \
        workloads/NeonLogicalAndWorkload.cpp \
        workloads/NeonLogicalNotWorkload.cpp \
        workloads/NeonLogicalOrWorkload.cpp \
        workloads/NeonLogSoftmaxWorkload.cpp \
        workloads/NeonLstmFloatWorkload.cpp \
        workloads/NeonMaximumWorkload.cpp \
        workloads/NeonMeanWorkload.cpp \
        workloads/NeonMinimumWorkload.cpp \
        workloads/NeonMultiplicationWorkload.cpp \
        workloads/NeonDivisionWorkload.cpp \
        workloads/NeonNegWorkload.cpp \
        workloads/NeonNormalizationFloatWorkload.cpp \
        workloads/NeonPadWorkload.cpp \
        workloads/NeonPermuteWorkload.cpp \
        workloads/NeonPooling2dWorkload.cpp \
        workloads/NeonPreluWorkload.cpp \
        workloads/NeonQLstmWorkload.cpp \
        workloads/NeonQuantizedLstmWorkload.cpp \
        workloads/NeonQuantizeWorkload.cpp \
        workloads/NeonReduceWorkload.cpp \
        workloads/NeonReshapeWorkload.cpp \
        workloads/NeonResizeWorkload.cpp \
        workloads/NeonRsqrtWorkload.cpp \
        workloads/NeonSinWorkload.cpp \
        workloads/NeonSliceWorkload.cpp \
        workloads/NeonSoftmaxWorkload.cpp \
        workloads/NeonSpaceToBatchNdWorkload.cpp \
        workloads/NeonSpaceToDepthWorkload.cpp \
        workloads/NeonSplitterWorkload.cpp \
        workloads/NeonStackWorkload.cpp \
        workloads/NeonStridedSliceWorkload.cpp \
        workloads/NeonSubtractionWorkload.cpp \
        workloads/NeonTransposeConvolution2dWorkload.cpp \
        workloads/NeonTransposeWorkload.cpp

else

# ARMNN_COMPUTE_NEON_ENABLED == 0
# No source file will be compiled for the NEON backend

BACKEND_SOURCES :=

endif

# BACKEND_TEST_SOURCES contains the list of files to be included
# in the Android unit test build (armnn-tests) and it is picked
# up by the Android.mk file in the root of ArmNN

# The variable to enable/disable the NEON backend (ARMNN_COMPUTE_NEON_ENABLED is declared in android-nn-driver/Android.mk)
ifeq ($(ARMNN_COMPUTE_NEON_ENABLED),1)

# ARMNN_COMPUTE_NEON_ENABLED == 1
# Include the source files for the NEON backend tests

BACKEND_TEST_SOURCES := \
        test/NeonBackendTests.cpp \
        test/NeonCreateWorkloadTests.cpp \
        test/NeonEndToEndTests.cpp \
        test/NeonJsonPrinterTests.cpp \
        test/NeonLayerSupportTests.cpp \
        test/NeonLayerTests.cpp \
        test/NeonOptimizedNetworkTests.cpp \
        test/NeonRuntimeTests.cpp \
        test/NeonTimerTest.cpp

ifeq ($(ARMNN_REF_ENABLED),1)
BACKEND_TEST_SOURCES += \
        test/NeonMemCopyTests.cpp
endif # ARMNN_REF_ENABLED == 1

else

# ARMNN_COMPUTE_NEON_ENABLED == 0
# No source file will be compiled for the NEON backend tests

BACKEND_TEST_SOURCES :=

endif
