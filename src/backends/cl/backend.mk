#
# Copyright © 2017 ARM Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

# BACKEND_SOURCES contains the list of files to be included
# in the Android build and it is picked up by the Android.mk
# file in the root of ArmNN

# The variable to enable/disable the CL backend (ARMNN_COMPUTE_CL_ENABLED is declared in android-nn-driver/Android.mk)
ifeq ($(ARMNN_COMPUTE_CL_ENABLED),1)

# ARMNN_COMPUTE_CL_ENABLED == 1
# Include the source files for the CL backend

BACKEND_SOURCES := \
        ClBackend.cpp \
        ClBackendContext.cpp \
        ClBackendModelContext.cpp \
        ClContextControl.cpp \
        ClContextDeserializer.cpp \
        ClContextSerializer.cpp \
        ClImportTensorHandleFactory.cpp \
        ClLayerSupport.cpp \
        ClRegistryInitializer.cpp \
        ClTensorHandleFactory.cpp \
        ClWorkloadFactory.cpp \
        OpenClTimer.cpp \
        workloads/ClAbsWorkload.cpp \
        workloads/ClActivationWorkload.cpp \
        workloads/ClAdditionWorkload.cpp \
        workloads/ClArgMinMaxWorkload.cpp \
        workloads/ClBatchNormalizationFloatWorkload.cpp \
        workloads/ClBatchToSpaceNdWorkload.cpp \
        workloads/ClCastWorkload.cpp \
        workloads/ClChannelShuffleWorkload.cpp \
        workloads/ClComparisonWorkload.cpp \
        workloads/ClConcatWorkload.cpp \
        workloads/ClConstantWorkload.cpp \
        workloads/ClConvertFp16ToFp32Workload.cpp \
        workloads/ClConvertFp32ToFp16Workload.cpp \
        workloads/ClConvolution2dWorkload.cpp \
        workloads/ClConvolution3dWorkload.cpp \
        workloads/ClDepthToSpaceWorkload.cpp \
        workloads/ClDepthwiseConvolutionWorkload.cpp \
        workloads/ClDequantizeWorkload.cpp \
        workloads/ClDivisionWorkload.cpp \
        workloads/ClExpWorkload.cpp \
        workloads/ClFillWorkload.cpp \
        workloads/ClFloorFloatWorkload.cpp \
        workloads/ClFullyConnectedWorkload.cpp \
        workloads/ClGatherWorkload.cpp \
        workloads/ClInstanceNormalizationWorkload.cpp \
        workloads/ClL2NormalizationFloatWorkload.cpp \
        workloads/ClLogWorkload.cpp \
        workloads/ClLogicalAndWorkload.cpp \
        workloads/ClLogicalNotWorkload.cpp \
        workloads/ClLogicalOrWorkload.cpp \
        workloads/ClLogSoftmaxWorkload.cpp \
        workloads/ClLstmFloatWorkload.cpp \
        workloads/ClMaximumWorkload.cpp \
        workloads/ClMeanWorkload.cpp \
        workloads/ClMinimumWorkload.cpp \
        workloads/ClMultiplicationWorkload.cpp \
        workloads/ClNegWorkload.cpp \
        workloads/ClNormalizationFloatWorkload.cpp \
        workloads/ClPadWorkload.cpp \
        workloads/ClPermuteWorkload.cpp \
        workloads/ClPooling2dWorkload.cpp \
        workloads/ClPreluWorkload.cpp \
        workloads/ClQLstmWorkload.cpp \
        workloads/ClQuantizedLstmWorkload.cpp \
        workloads/ClQuantizeWorkload.cpp \
        workloads/ClReduceWorkload.cpp \
        workloads/ClReshapeWorkload.cpp \
        workloads/ClResizeWorkload.cpp \
        workloads/ClRsqrtWorkload.cpp \
        workloads/ClSinWorkload.cpp \
        workloads/ClSliceWorkload.cpp \
        workloads/ClSoftmaxWorkload.cpp \
        workloads/ClSpaceToBatchNdWorkload.cpp \
        workloads/ClSpaceToDepthWorkload.cpp \
        workloads/ClSplitterWorkload.cpp \
        workloads/ClStackWorkload.cpp \
        workloads/ClStridedSliceWorkload.cpp \
        workloads/ClSubtractionWorkload.cpp \
        workloads/ClTransposeConvolution2dWorkload.cpp \
        workloads/ClTransposeWorkload.cpp
else

# ARMNN_COMPUTE_CL_ENABLED == 0
# No source file will be compiled for the CL backend

BACKEND_SOURCES :=

endif

# BACKEND_TEST_SOURCES contains the list of files to be included
# in the Android unit test build (armnn-tests) and it is picked
# up by the Android.mk file in the root of ArmNN

# The variable to enable/disable the CL backend (ARMNN_COMPUTE_CL_ENABLED is declared in android-nn-driver/Android.mk)
ifeq ($(ARMNN_COMPUTE_CL_ENABLED),1)

# ARMNN_COMPUTE_CL_ENABLED == 1
# Include the source files for the CL backend tests

BACKEND_TEST_SOURCES := \
        test/ClBackendTests.cpp \
        test/ClContextSerializerTests.cpp \
        test/ClCreateWorkloadTests.cpp \
        test/ClEndToEndTests.cpp \
        test/ClJsonPrinterTests.cpp \
        test/ClLayerSupportTests.cpp \
        test/ClLayerTests.cpp \
        test/ClOptimizedNetworkTests.cpp \
        test/ClRuntimeTests.cpp \
        test/Fp16SupportTest.cpp \
        test/OpenClTimerTest.cpp

ifeq ($(ARMNN_REF_ENABLED),1)
BACKEND_TEST_SOURCES += \
        test/ClMemCopyTests.cpp
endif # ARMNN_REF_ENABLED == 1

else

# ARMNN_COMPUTE_CL_ENABLED == 0
# No source file will be compiled for the CL backend tests

BACKEND_TEST_SOURCES :=

endif
