#
# Copyright © 2017 ARM Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

# BACKEND_SOURCES contains the list of files to be included
# in the Android build and it is picked up by the Android.mk
# file in the root of ArmNN

BACKEND_SOURCES := \
        ClBackend.cpp \
        ClContextControl.cpp \
        ClLayerSupport.cpp \
        ClWorkloadFactory.cpp \
        workloads/ClActivationFloatWorkload.cpp \
        workloads/ClActivationUint8Workload.cpp \
        workloads/ClAdditionWorkload.cpp \
        workloads/ClBaseConstantWorkload.cpp \
        workloads/ClBatchNormalizationFloatWorkload.cpp \
        workloads/ClConstantFloatWorkload.cpp \
        workloads/ClConstantUint8Workload.cpp \
        workloads/ClConvertFp16ToFp32Workload.cpp \
        workloads/ClConvertFp32ToFp16Workload.cpp \
        workloads/ClConvolution2dBaseWorkload.cpp \
        workloads/ClConvolution2dFloatWorkload.cpp \
        workloads/ClConvolution2dUint8Workload.cpp \
        workloads/ClDepthwiseConvolutionBaseWorkload.cpp \
        workloads/ClDepthwiseConvolutionFloatWorkload.cpp \
        workloads/ClDepthwiseConvolutionUint8Workload.cpp \
        workloads/ClDivisionFloatWorkload.cpp \
        workloads/ClFloorFloatWorkload.cpp \
        workloads/ClFullyConnectedWorkload.cpp \
        workloads/ClL2NormalizationFloatWorkload.cpp \
        workloads/ClLstmFloatWorkload.cpp \
        workloads/ClMergerFloatWorkload.cpp \
        workloads/ClMergerUint8Workload.cpp \
        workloads/ClMultiplicationWorkload.cpp \
        workloads/ClNormalizationFloatWorkload.cpp \
        workloads/ClPadWorkload.cpp \
        workloads/ClPermuteWorkload.cpp \
        workloads/ClPooling2dBaseWorkload.cpp \
        workloads/ClPooling2dFloatWorkload.cpp \
        workloads/ClPooling2dUint8Workload.cpp \
        workloads/ClReshapeFloatWorkload.cpp \
        workloads/ClReshapeUint8Workload.cpp \
        workloads/ClResizeBilinearFloatWorkload.cpp \
        workloads/ClSoftmaxBaseWorkload.cpp \
        workloads/ClSoftmaxFloatWorkload.cpp \
        workloads/ClSoftmaxUint8Workload.cpp \
        workloads/ClSubtractionWorkload.cpp
