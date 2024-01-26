#
# Copyright © 2023-2024 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

list(APPEND armnnGpuFsaBackendLayerValidators_sources
        GpuFsaConvolution2d.cpp
        GpuFsaConvolution2d.hpp
        GpuFsaDepthwiseConvolution2d.cpp
        GpuFsaDepthwiseConvolution2d.hpp
        GpuFsaElementwiseBinaryAdd.cpp
        GpuFsaElementwiseBinaryAdd.hpp
    )

add_library(armnnGpuFsaBackendLayerValidators OBJECT ${armnnGpuFsaBackendLayerValidators_sources})
target_include_directories(armnnGpuFsaBackendLayerValidators PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnGpuFsaBackendLayerValidators PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
target_include_directories(armnnGpuFsaBackendLayerValidators PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
