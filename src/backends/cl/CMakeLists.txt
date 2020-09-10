#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

if(ARMCOMPUTECL)
    list(APPEND armnnClBackend_sources
        ClBackend.cpp
        ClBackend.hpp
        ClBackendContext.cpp
        ClBackendContext.hpp
        ClBackendId.hpp
        ClBackendModelContext.cpp
        ClBackendModelContext.hpp
        ClContextControl.cpp
        ClContextControl.hpp
        ClLayerSupport.cpp
        ClLayerSupport.hpp
        ClRegistryInitializer.cpp
        ClTensorHandle.hpp
        ClTensorHandleFactory.cpp
        ClTensorHandleFactory.hpp
        ClWorkloadFactory.cpp
        ClWorkloadFactory.hpp
        OpenClTimer.cpp
        OpenClTimer.hpp
    )

    add_subdirectory(workloads)

    if(BUILD_UNIT_TESTS)
        add_subdirectory(test)
    endif()

else()
    list(APPEND armnnClBackend_sources
        ClBackendId.hpp
        ClLayerSupport.cpp
        ClLayerSupport.hpp
    )
endif()

add_library(armnnClBackend OBJECT ${armnnClBackend_sources})
target_include_directories(armnnClBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnClBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
target_include_directories(armnnClBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
target_include_directories(armnnClBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/profiling)
target_include_directories(armnnClBackend PRIVATE ${PROJECT_SOURCE_DIR}/profiling/common/include)
