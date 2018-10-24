#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

list(APPEND armnnNeonBackend_sources
    NeonBackendId.hpp
    NeonLayerSupport.cpp
    NeonLayerSupport.hpp
    NeonWorkloadFactory.cpp
    NeonWorkloadFactory.hpp
)

if(ARMCOMPUTENEON)
    list(APPEND armnnNeonBackend_sources
        NeonBackend.cpp
        NeonBackend.hpp
        NeonInterceptorScheduler.hpp
        NeonInterceptorScheduler.cpp
        NeonTensorHandle.hpp
        NeonTimer.hpp
        NeonTimer.cpp
    )

    add_subdirectory(workloads)

    if(BUILD_UNIT_TESTS)
        add_subdirectory(test)
    endif()
endif()

add_library(armnnNeonBackend OBJECT ${armnnNeonBackend_sources})
target_include_directories(armnnNeonBackend PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(armnnNeonBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnNeonBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)

