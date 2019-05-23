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
        workloads/BatchToSpaceNd.cpp \
        workloads/Broadcast.cpp \
        workloads/ConvImpl.cpp \
        workloads/Debug.cpp \
        workloads/DetectionPostProcess.cpp \
        workloads/ElementwiseFunction.cpp \
        workloads/FullyConnected.cpp \
        workloads/Gather.cpp \
        workloads/Mean.cpp \
        workloads/Merger.cpp \
        workloads/Pad.cpp \
        workloads/Pooling2d.cpp \
        workloads/RefActivationWorkload.cpp \
        workloads/RefBatchNormalizationFloat32Workload.cpp \
        workloads/RefBatchNormalizationUint8Workload.cpp \
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
        workloads/RefDetectionPostProcessFloat32Workload.cpp \
        workloads/RefDetectionPostProcessUint8Workload.cpp \
        workloads/RefElementwiseWorkload.cpp \
        workloads/RefFakeQuantizationFloat32Workload.cpp \
        workloads/RefFloorFloat32Workload.cpp \
        workloads/RefFullyConnectedFloat32Workload.cpp \
        workloads/RefFullyConnectedUint8Workload.cpp \
        workloads/RefGatherWorkload.cpp \
        workloads/RefL2NormalizationFloat32Workload.cpp \
        workloads/RefLstmWorkload.cpp \
        workloads/RefMeanFloat32Workload.cpp \
        workloads/RefMeanUint8Workload.cpp \
        workloads/RefNormalizationFloat32Workload.cpp \
        workloads/RefPadWorkload.cpp \
        workloads/RefPermuteWorkload.cpp \
        workloads/RefPooling2dFloat32Workload.cpp \
        workloads/RefPooling2dUint8Workload.cpp \
        workloads/RefQuantizeWorkload.cpp \
        workloads/RefReshapeFloat32Workload.cpp \
        workloads/RefReshapeUint8Workload.cpp \
        workloads/RefResizeBilinearFloat32Workload.cpp \
        workloads/RefResizeBilinearUint8Workload.cpp \
        workloads/RefRsqrtFloat32Workload.cpp \
        workloads/RefSoftmaxFloat32Workload.cpp \
        workloads/RefSoftmaxUint8Workload.cpp \
        workloads/RefSpaceToBatchNdWorkload.cpp \
        workloads/RefStridedSliceWorkload.cpp \
        workloads/RefSplitterFloat32Workload.cpp \
        workloads/RefSplitterUint8Workload.cpp \
        workloads/ResizeBilinear.cpp \
        workloads/Rsqrt.cpp \
        workloads/SpaceToBatchNd.cpp \
        workloads/StridedSlice.cpp \
        workloads/StringMapping.cpp \
        workloads/Softmax.cpp

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
