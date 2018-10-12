#
# Copyright © 2017 ARM Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

# BACKEND_SOURCES contains the list of files to be included
# in the Android build and it is picked up by the Android.mk
# file in the root of ArmNN

BACKEND_SOURCES := \
        NeonBackend.cpp \
        NeonLayerSupport.cpp \
        NeonWorkloadFactory.cpp \
        workloads/NeonActivationWorkload.cpp \
        workloads/NeonAdditionFloatWorkload.cpp \
        workloads/NeonBatchNormalizationFloatWorkload.cpp \
        workloads/NeonConstantFloatWorkload.cpp \
        workloads/NeonConstantUint8Workload.cpp \
        workloads/NeonConvertFp16ToFp32Workload.cpp \
        workloads/NeonConvertFp32ToFp16Workload.cpp \
        workloads/NeonConvolution2dBaseWorkload.cpp \
        workloads/NeonConvolution2dFloatWorkload.cpp \
        workloads/NeonConvolution2dUint8Workload.cpp \
        workloads/NeonDepthwiseConvolutionBaseWorkload.cpp \
        workloads/NeonDepthwiseConvolutionFloatWorkload.cpp \
        workloads/NeonDepthwiseConvolutionUint8Workload.cpp \
        workloads/NeonFloorFloatWorkload.cpp \
        workloads/NeonFullyConnectedWorkload.cpp \
        workloads/NeonL2NormalizationFloatWorkload.cpp \
        workloads/NeonLstmFloatWorkload.cpp \
        workloads/NeonMergerFloatWorkload.cpp \
        workloads/NeonMergerUint8Workload.cpp \
        workloads/NeonMultiplicationFloatWorkload.cpp \
        workloads/NeonNormalizationFloatWorkload.cpp \
        workloads/NeonPermuteWorkload.cpp \
        workloads/NeonPooling2dBaseWorkload.cpp \
        workloads/NeonPooling2dFloatWorkload.cpp \
        workloads/NeonPooling2dUint8Workload.cpp \
        workloads/NeonReshapeFloatWorkload.cpp \
        workloads/NeonReshapeUint8Workload.cpp \
        workloads/NeonSoftmaxBaseWorkload.cpp \
        workloads/NeonSoftmaxFloatWorkload.cpp \
        workloads/NeonSoftmaxUint8Workload.cpp \
        workloads/NeonSplitterFloatWorkload.cpp \
        workloads/NeonSplitterUint8Workload.cpp \
        workloads/NeonSubtractionFloatWorkload.cpp \
        workloads/NeonWorkloadUtils.cpp
