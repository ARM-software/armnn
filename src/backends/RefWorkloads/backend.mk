#
# Copyright © 2017 ARM Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

BACKEND_SOURCES := \
        FullyConnected.cpp \
        RefMergerUint8Workload.cpp \
        RefResizeBilinearUint8Workload.cpp \
        RefFullyConnectedFloat32Workload.cpp \
        RefSoftmaxFloat32Workload.cpp \
        RefActivationFloat32Workload.cpp \
        RefBatchNormalizationUint8Workload.cpp \
        RefBaseConstantWorkload.cpp \
        RefResizeBilinearFloat32Workload.cpp \
        RefBatchNormalizationFloat32Workload.cpp \
        Broadcast.cpp \
        ArithmeticFunction.cpp \
        RefArithmeticWorkload.cpp \
        RefFakeQuantizationFloat32Workload.cpp \
        ResizeBilinear.cpp \
        RefSoftmaxUint8Workload.cpp \
        RefDepthwiseConvolution2dFloat32Workload.cpp \
        RefPooling2dUint8Workload.cpp \
        RefFloorFloat32Workload.cpp \
        ConvImpl.cpp \
        Activation.cpp \
        RefReshapeUint8Workload.cpp \
        RefL2NormalizationFloat32Workload.cpp \
        RefLstmFloat32Workload.cpp \
        RefConvolution2dFloat32Workload.cpp \
        RefConvolution2dUint8Workload.cpp \
        RefSplitterFloat32Workload.cpp \
        RefActivationUint8Workload.cpp \
        RefSplitterUint8Workload.cpp \
        RefPooling2dFloat32Workload.cpp \
        RefReshapeFloat32Workload.cpp \
        RefNormalizationFloat32Workload.cpp \
        Softmax.cpp \
        RefDepthwiseConvolution2dUint8Workload.cpp \
        RefConstantUint8Workload.cpp \
        RefConstantFloat32Workload.cpp \
        Pooling2d.cpp \
        RefMergerFloat32Workload.cpp \
        RefFullyConnectedUint8Workload.cpp \
        RefPermuteWorkload.cpp \
        RefConvertFp16ToFp32Workload.cpp \
        RefConvertFp32ToFp16Workload.cpp

