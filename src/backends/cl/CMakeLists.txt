#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

list(APPEND armnnClBackend_sources
    ClBackendId.hpp
    ClLayerSupport.cpp
    ClLayerSupport.hpp
    ClWorkloadFactory.cpp
    ClWorkloadFactory.hpp
)

if(ARMCOMPUTECL)
    list(APPEND armnnClBackend_sources
        ClBackendContext.cpp
        ClBackendContext.hpp
        ClBackend.cpp
        ClBackend.hpp
        ClContextControl.cpp
        ClContextControl.hpp
        ClTensorHandle.hpp
        OpenClTimer.cpp
        OpenClTimer.hpp
    )

    add_subdirectory(workloads)

    if(BUILD_UNIT_TESTS)
        add_subdirectory(test)
    endif()
endif()

add_library(armnnClBackend OBJECT ${armnnClBackend_sources})
target_include_directories(armnnClBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnClBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
target_include_directories(armnnClBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)

