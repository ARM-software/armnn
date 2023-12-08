#
# Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

if(ARMCOMPUTEGPUFSA)
    list(APPEND armnnGpuFsaBackend_sources
        GpuFsaBackend.cpp
        GpuFsaBackend.hpp
        GpuFsaBackendContext.cpp
        GpuFsaBackendContext.hpp
        GpuFsaBackendDefaultAllocator.hpp
        GpuFsaBackendId.hpp
        GpuFsaContextControl.cpp
        GpuFsaContextControl.hpp
        GpuFsaLayerSupport.cpp
        GpuFsaLayerSupport.hpp
        GpuFsaRegistryInitializer.cpp
        GpuFsaTensorHandle.hpp
        GpuFsaTensorHandleFactory.cpp
        GpuFsaTensorHandleFactory.hpp
        GpuFsaWorkloadFactory.cpp
        GpuFsaWorkloadFactory.hpp
    )

    add_subdirectory(layers)
    add_subdirectory(workloads)

    if(BUILD_UNIT_TESTS)
        add_subdirectory(test)
    endif()

else()
    list(APPEND armnnGpuFsaBackend_sources
        GpuFsaBackendId.hpp
        GpuFsaLayerSupport.cpp
        GpuFsaLayerSupport.hpp
    )
endif()

add_library(armnnGpuFsaBackend OBJECT ${armnnGpuFsaBackend_sources})
target_include_directories(armnnGpuFsaBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnGpuFsaBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
target_include_directories(armnnGpuFsaBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)