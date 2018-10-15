#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

list(APPEND armnnNeonBackendWorkloads_sources
    NeonActivationWorkload.cpp
    NeonActivationWorkload.hpp
    NeonAdditionFloatWorkload.cpp
    NeonAdditionFloatWorkload.hpp
    NeonBatchNormalizationFloatWorkload.cpp
    NeonBatchNormalizationFloatWorkload.hpp
    NeonConstantWorkload.cpp
    NeonConstantWorkload.hpp
    NeonConvertFp16ToFp32Workload.cpp
    NeonConvertFp16ToFp32Workload.hpp
    NeonConvertFp32ToFp16Workload.cpp
    NeonConvertFp32ToFp16Workload.hpp
    NeonConvolution2dWorkload.cpp
    NeonConvolution2dWorkload.hpp
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
    NeonPooling2dWorkload.cpp
    NeonPooling2dWorkload.hpp
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
    NeonWorkloadUtils.hpp
)

add_library(armnnNeonBackendWorkloads OBJECT ${armnnNeonBackendWorkloads_sources})
target_include_directories(armnnNeonBackendWorkloads PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(armnnNeonBackendWorkloads PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnNeonBackendWorkloads PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
