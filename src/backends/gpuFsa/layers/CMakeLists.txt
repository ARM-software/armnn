#
# Copyright © 2023-2024 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

list(APPEND armnnGpuFsaBackendLayers_sources
        GpuFsaCast.cpp
        GpuFsaCast.hpp
        GpuFsaConvolution2d.cpp
        GpuFsaConvolution2d.hpp
        GpuFsaDepthwiseConvolution2d.cpp
        GpuFsaDepthwiseConvolution2d.hpp
        GpuFsaElementwiseBinaryAdd.cpp
        GpuFsaElementwiseBinaryAdd.hpp
        GpuFsaElementwiseBinarySub.cpp
        GpuFsaElementwiseBinarySub.hpp
        GpuFsaPooling2d.cpp
        GpuFsaPooling2d.hpp
        UtilsGpuFsa.cpp
        UtilsGpuFsa.hpp
    )

add_library(armnnGpuFsaBackendLayers OBJECT ${armnnGpuFsaBackendLayers_sources})
target_include_directories(armnnGpuFsaBackendLayers PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnGpuFsaBackendLayers PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
target_include_directories(armnnGpuFsaBackendLayers PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)