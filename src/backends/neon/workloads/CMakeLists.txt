#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

list(APPEND armnnNeonBackendWorkloads_sources
    NeonActivationWorkload.cpp
    NeonActivationWorkload.hpp
    NeonAdditionFloatWorkload.cpp
    NeonAdditionFloatWorkload.hpp
    NeonBaseConstantWorkload.hpp
    NeonBatchNormalizationFloatWorkload.cpp
    NeonBatchNormalizationFloatWorkload.hpp
    NeonConstantFloatWorkload.cpp
    NeonConstantFloatWorkload.hpp
    NeonConstantUint8Workload.cpp
    NeonConstantUint8Workload.hpp
    NeonConvertFp16ToFp32Workload.cpp
    NeonConvertFp16ToFp32Workload.hpp
    NeonConvertFp32ToFp16Workload.cpp
    NeonConvertFp32ToFp16Workload.hpp
    NeonConvolution2dBaseWorkload.cpp
    NeonConvolution2dBaseWorkload.hpp
    NeonConvolution2dFloatWorkload.cpp
    NeonConvolution2dFloatWorkload.hpp
    NeonConvolution2dUint8Workload.cpp
    NeonConvolution2dUint8Workload.hpp
    NeonDepthwiseConvolutionBaseWorkload.cpp
    NeonDepthwiseConvolutionBaseWorkload.hpp
    NeonDepthwiseConvolutionFloatWorkload.cpp
    NeonDepthwiseConvolutionFloatWorkload.hpp
    NeonDepthwiseConvolutionUint8Workload.cpp
    NeonDepthwiseConvolutionUint8Workload.hpp
    NeonFloorFloatWorkload.cpp
    NeonFloorFloatWorkload.hpp
    NeonFullyConnectedWorkload.cpp
    NeonFullyConnectedWorkload.hpp
    NeonL2NormalizationFloatWorkload.cpp
    NeonL2NormalizationFloatWorkload.hpp
    NeonLstmFloatWorkload.cpp
    NeonLstmFloatWorkload.hpp
    NeonMergerWorkload.hpp
    NeonMultiplicationFloatWorkload.cpp
    NeonMultiplicationFloatWorkload.hpp
    NeonNormalizationFloatWorkload.cpp
    NeonNormalizationFloatWorkload.hpp
    NeonPermuteWorkload.cpp
    NeonPermuteWorkload.hpp
    NeonPooling2dBaseWorkload.cpp
    NeonPooling2dBaseWorkload.hpp
    NeonPooling2dFloatWorkload.cpp
    NeonPooling2dFloatWorkload.hpp
    NeonPooling2dUint8Workload.cpp
    NeonPooling2dUint8Workload.hpp
    NeonReshapeWorkload.cpp
    NeonReshapeWorkload.hpp
    NeonSoftmaxBaseWorkload.cpp
    NeonSoftmaxBaseWorkload.hpp
    NeonSoftmaxFloatWorkload.cpp
    NeonSoftmaxFloatWorkload.hpp
    NeonSoftmaxUint8Workload.cpp
    NeonSoftmaxUint8Workload.hpp
    NeonSplitterWorkload.hpp
    NeonSubtractionFloatWorkload.cpp
    NeonSubtractionFloatWorkload.hpp
    NeonWorkloads.hpp
    NeonWorkloadUtils.cpp
    NeonWorkloadUtils.hpp
)

add_library(armnnNeonBackendWorkloads STATIC ${armnnNeonBackendWorkloads_sources})
target_include_directories(armnnNeonBackendWorkloads PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(armnnNeonBackendWorkloads PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnNeonBackendWorkloads PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
