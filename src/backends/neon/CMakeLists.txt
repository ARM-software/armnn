#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

if(ARMCOMPUTENEON)
    list(APPEND armnnNeonBackend_sources
        NeonBackend.cpp
        NeonBackend.hpp
        NeonLayerSupport.cpp
        NeonLayerSupport.hpp
        NeonWorkloadFactory.cpp
        NeonWorkloadFactory.hpp
        NeonTensorHandle.hpp
    )

    add_subdirectory(workloads)
    add_subdirectory(test)
else()
    list(APPEND armnnNeonBackend_sources
        NeonLayerSupport.cpp
        NeonLayerSupport.hpp
        NeonWorkloadFactory.cpp
        NeonWorkloadFactory.hpp
    )
endif()

add_library(armnnNeonBackend STATIC ${armnnNeonBackend_sources})
target_include_directories(armnnNeonBackend PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(armnnNeonBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnNeonBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
