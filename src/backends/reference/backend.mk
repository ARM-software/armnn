#
# Copyright © 2017 ARM Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

# BACKEND_SOURCES contains the list of files to be included
# in the Android build and it is picked up by the Android.mk
# file in the root of ArmNN

BACKEND_SOURCES := \
        RefBackend.cpp \
        RefLayerSupport.cpp \
        RefWorkloadFactory.cpp \
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
        workloads/Mean.cpp \
        workloads/Concatenate.cpp \
        workloads/Pad.cpp \
        workloads/Pooling2d.cpp \
        workloads/PreluImpl.cpp \
        workloads/RefActivationWorkload.cpp \
        workloads/RefBatchNormalizationWorkload.cpp \
        workloads/RefBatchToSpaceNdFloat32Workload.cpp \
        workloads/RefBatchToSpaceNdUint8Workload.cpp \
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
        workloads/RefRsqrtWorkload.cpp \
        workloads/RefSoftmaxWorkload.cpp \
        workloads/RefSpaceToBatchNdWorkload.cpp \
        workloads/RefSpaceToDepthWorkload.cpp \
        workloads/RefStridedSliceWorkload.cpp \
        workloads/RefSplitterWorkload.cpp \
        workloads/ResizeBilinear.cpp \
        workloads/Rsqrt.cpp \
        workloads/SpaceToBatchNd.cpp \
        workloads/SpaceToDepth.cpp \
        workloads/StridedSlice.cpp \
        workloads/StringMapping.cpp \
        workloads/Softmax.cpp \
        workloads/Splitter.cpp

# BACKEND_TEST_SOURCES contains the list of files to be included
# in the Android unit test build (armnn-tests) and it is picked
# up by the Android.mk file in the root of ArmNN

BACKEND_TEST_SOURCES := \
        test/RefCreateWorkloadTests.cpp \
        test/RefEndToEndTests.cpp \
        test/RefJsonPrinterTests.cpp \
        test/RefLayerSupportTests.cpp \
        test/RefLayerTests.cpp \
        test/RefOptimizedNetworkTests.cpp \
        test/RefRuntimeTests.cpp
