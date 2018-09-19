#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

list(APPEND armnnClBackend_sources
    ClActivationFloatWorkload.cpp
    ClActivationFloatWorkload.hpp
    ClActivationUint8Workload.cpp
    ClActivationUint8Workload.hpp
    ClAdditionWorkload.cpp
    ClAdditionWorkload.hpp
    ClBaseConstantWorkload.cpp
    ClBaseConstantWorkload.hpp
    ClBaseMergerWorkload.hpp
    ClBaseSplitterWorkload.hpp
    ClBatchNormalizationFloatWorkload.cpp
    ClBatchNormalizationFloatWorkload.hpp
    ClConstantFloatWorkload.cpp
    ClConstantFloatWorkload.hpp
    ClConstantUint8Workload.cpp
    ClConstantUint8Workload.hpp
    ClConvertFp16ToFp32Workload.cpp
    ClConvertFp16ToFp32Workload.hpp
    ClConvertFp32ToFp16Workload.cpp
    ClConvertFp32ToFp16Workload.hpp
    ClConvolution2dBaseWorkload.cpp
    ClConvolution2dBaseWorkload.hpp
    ClConvolution2dFloatWorkload.cpp
    ClConvolution2dFloatWorkload.hpp
    ClConvolution2dUint8Workload.cpp
    ClConvolution2dUint8Workload.hpp
    ClDepthwiseConvolutionBaseWorkload.cpp
    ClDepthwiseConvolutionBaseWorkload.hpp
    ClDepthwiseConvolutionFloatWorkload.cpp
    ClDepthwiseConvolutionFloatWorkload.hpp
    ClDepthwiseConvolutionUint8Workload.cpp
    ClDepthwiseConvolutionUint8Workload.hpp
    ClDivisionFloatWorkload.cpp
    ClDivisionFloatWorkload.hpp
    ClFloorFloatWorkload.cpp
    ClFloorFloatWorkload.hpp
    ClFullyConnectedWorkload.cpp
    ClFullyConnectedWorkload.hpp
    ClL2NormalizationFloatWorkload.cpp
    ClL2NormalizationFloatWorkload.hpp
    ClLstmFloatWorkload.cpp
    ClLstmFloatWorkload.hpp
    ClMergerFloatWorkload.cpp
    ClMergerFloatWorkload.hpp
    ClMergerUint8Workload.cpp
    ClMergerUint8Workload.hpp
    ClMultiplicationFloatWorkload.cpp
    ClMultiplicationFloatWorkload.hpp
    ClNormalizationFloatWorkload.cpp
    ClNormalizationFloatWorkload.hpp
    ClPermuteWorkload.cpp
    ClPermuteWorkload.hpp
    ClPooling2dBaseWorkload.cpp
    ClPooling2dBaseWorkload.hpp
    ClPooling2dFloatWorkload.cpp
    ClPooling2dFloatWorkload.hpp
    ClPooling2dUint8Workload.cpp
    ClPooling2dUint8Workload.hpp
    ClReshapeFloatWorkload.cpp
    ClReshapeFloatWorkload.hpp
    ClReshapeUint8Workload.cpp
    ClReshapeUint8Workload.hpp
    ClResizeBilinearFloatWorkload.cpp
    ClResizeBilinearFloatWorkload.hpp
    ClSoftmaxBaseWorkload.cpp
    ClSoftmaxBaseWorkload.hpp
    ClSoftmaxFloatWorkload.cpp
    ClSoftmaxFloatWorkload.hpp
    ClSoftmaxUint8Workload.cpp
    ClSoftmaxUint8Workload.hpp
    ClSplitterFloatWorkload.cpp
    ClSplitterFloatWorkload.hpp
    ClSplitterUint8Workload.cpp
    ClSplitterUint8Workload.hpp
    ClSubtractionWorkload.cpp
    ClSubtractionWorkload.hpp
    ClWorkloadUtils.hpp
)

add_library(armnnClBackend STATIC ${armnnClBackend_sources})
target_include_directories(armnnClBackend PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(armnnClBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnClBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
