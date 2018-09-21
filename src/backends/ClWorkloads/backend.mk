#
# Copyright © 2017 ARM Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

# BACKEND_SOURCES contains the list of files to be included
# in the Android build and it is picked up by the Android.mk
# file in the root of ArmNN

BACKEND_SOURCES := \
        ClActivationFloatWorkload.cpp \
        ClActivationUint8Workload.cpp \
        ClAdditionWorkload.cpp \
        ClSubtractionWorkload.cpp \
        ClBaseConstantWorkload.cpp \
        ClBatchNormalizationFloatWorkload.cpp \
        ClConstantFloatWorkload.cpp \
        ClConstantUint8Workload.cpp \
        ClConvertFp16ToFp32Workload.cpp \
        ClConvertFp32ToFp16Workload.cpp \
        ClConvolution2dBaseWorkload.cpp \
        ClConvolution2dFloatWorkload.cpp \
        ClConvolution2dUint8Workload.cpp \
        ClDepthwiseConvolutionBaseWorkload.cpp \
        ClDepthwiseConvolutionFloatWorkload.cpp \
        ClDepthwiseConvolutionUint8Workload.cpp \
        ClDivisionFloatWorkload.cpp \
        ClFloorFloatWorkload.cpp \
        ClFullyConnectedWorkload.cpp \
        ClL2NormalizationFloatWorkload.cpp \
        ClLstmFloatWorkload.cpp \
        ClMergerFloatWorkload.cpp \
        ClMergerUint8Workload.cpp \
        ClMultiplicationFloatWorkload.cpp \
        ClNormalizationFloatWorkload.cpp \
        ClPadWorkload.cpp \
        ClPermuteWorkload.cpp \
        ClPooling2dBaseWorkload.cpp \
        ClPooling2dFloatWorkload.cpp \
        ClPooling2dUint8Workload.cpp \
        ClReshapeFloatWorkload.cpp \
        ClReshapeUint8Workload.cpp \
        ClResizeBilinearFloatWorkload.cpp \
        ClSoftmaxBaseWorkload.cpp \
        ClSoftmaxFloatWorkload.cpp \
        ClSoftmaxUint8Workload.cpp \
        ClSplitterFloatWorkload.cpp \
        ClSplitterUint8Workload.cpp

