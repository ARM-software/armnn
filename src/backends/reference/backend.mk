#
# Copyright © 2017 ARM Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

# BACKEND_SOURCES contains the list of files to be included
# in the Android build and it is picked up by the Android.mk
# file in the root of ArmNN

# The variable to enable/disable the reference backend (ARMNN_REF_ENABLED is declared in android-nn-driver/Android.mk)
ifeq ($(ARMNN_REF_ENABLED),1)

# ARMNN_REF_ENABLED == 1
# Include the source files for the reference backend

BACKEND_SOURCES := \
        RefBackend.cpp \
        RefLayerSupport.cpp \
        RefMemoryManager.cpp \
        RefTensorHandle.cpp \
        RefWorkloadFactory.cpp \
        RefRegistryInitializer.cpp \
        RefTensorHandleFactory.cpp \
        workloads/Activation.cpp \
        workloads/BatchNormImpl.cpp \
        workloads/BatchToSpaceNd.cpp \
        workloads/Broadcast.cpp \
        workloads/ConvImpl.cpp \
        workloads/Debug.cpp \
        workloads/DetectionPostProcess.cpp \
        workloads/ElementwiseFunction.cpp \
        workloads/FullyConnected.cpp \
        workloads/Gather.cpp \
        workloads/LstmUtils.cpp \
        workloads/Mean.cpp \
        workloads/Concatenate.cpp \
        workloads/Pad.cpp \
        workloads/Pooling2d.cpp \
        workloads/PreluImpl.cpp \
        workloads/RefActivationWorkload.cpp \
        workloads/RefBatchNormalizationWorkload.cpp \
        workloads/RefBatchToSpaceNdWorkload.cpp \
        workloads/RefConcatWorkload.cpp \
        workloads/RefConstantWorkload.cpp \
        workloads/RefConvertFp16ToFp32Workload.cpp \
        workloads/RefConvertFp32ToFp16Workload.cpp \
        workloads/RefConvolution2dWorkload.cpp \
        workloads/RefDebugWorkload.cpp \
        workloads/RefDepthwiseConvolution2dWorkload.cpp \
        workloads/RefDequantizeWorkload.cpp \
        workloads/RefDetectionPostProcessWorkload.cpp \
        workloads/RefElementwiseWorkload.cpp \
        workloads/RefFakeQuantizationFloat32Workload.cpp \
        workloads/RefFloorWorkload.cpp \
        workloads/RefFullyConnectedWorkload.cpp \
        workloads/RefGatherWorkload.cpp \
        workloads/RefL2NormalizationWorkload.cpp \
        workloads/RefLstmWorkload.cpp \
        workloads/RefMeanWorkload.cpp \
        workloads/RefNormalizationWorkload.cpp \
        workloads/RefPadWorkload.cpp \
        workloads/RefPermuteWorkload.cpp \
        workloads/RefPooling2dWorkload.cpp \
        workloads/RefPreluWorkload.cpp \
        workloads/RefQuantizeWorkload.cpp \
        workloads/RefReshapeWorkload.cpp \
        workloads/RefResizeBilinearWorkload.cpp \
        workloads/RefResizeWorkload.cpp \
        workloads/RefRsqrtWorkload.cpp \
        workloads/RefSoftmaxWorkload.cpp \
        workloads/RefSpaceToBatchNdWorkload.cpp \
        workloads/RefSpaceToDepthWorkload.cpp \
        workloads/RefStackWorkload.cpp \
        workloads/RefStridedSliceWorkload.cpp \
        workloads/RefSplitterWorkload.cpp \
        workloads/RefTransposeConvolution2dWorkload.cpp \
        workloads/Resize.cpp \
        workloads/Rsqrt.cpp \
        workloads/SpaceToBatchNd.cpp \
        workloads/SpaceToDepth.cpp \
        workloads/Stack.cpp \
        workloads/StridedSlice.cpp \
        workloads/StringMapping.cpp \
        workloads/Softmax.cpp \
        workloads/Splitter.cpp \
        workloads/TransposeConvolution2d.cpp
else

# ARMNN_REF_ENABLED == 0
# No source file will be compiled for the reference backend

BACKEND_SOURCES :=

endif

# BACKEND_TEST_SOURCES contains the list of files to be included
# in the Android unit test build (armnn-tests) and it is picked
# up by the Android.mk file in the root of ArmNN

# The variable to enable/disable the CL backend (ARMNN_REF_ENABLED is declared in android-nn-driver/Android.mk)
ifeq ($(ARMNN_REF_ENABLED),1)

# ARMNN_REF_ENABLED == 1
# Include the source files for the CL backend tests

BACKEND_TEST_SOURCES := \
        test/RefCreateWorkloadTests.cpp \
        test/RefDetectionPostProcessTests.cpp \
        test/RefEndToEndTests.cpp \
        test/RefJsonPrinterTests.cpp \
        test/RefLayerSupportTests.cpp \
        test/RefLayerTests.cpp \
        test/RefMemoryManagerTests.cpp \
        test/RefOptimizedNetworkTests.cpp \
        test/RefRuntimeTests.cpp
else

# ARMNN_REF_ENABLED == 0
# No source file will be compiled for the reference backend tests

BACKEND_TEST_SOURCES :=

endif
