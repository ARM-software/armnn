#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

if(ARMNNREF)
    list(APPEND armnnRefBackend_sources
        RefBackend.cpp
        RefBackend.hpp
        RefBackendId.hpp
        RefTensorHandle.hpp
        RefTensorHandle.cpp
        RefLayerSupport.cpp
        RefLayerSupport.hpp
        RefMemoryManager.hpp
        RefMemoryManager.cpp
        RefRegistryInitializer.cpp
        RefWorkloadFactory.cpp
        RefWorkloadFactory.hpp
        RefTensorHandleFactory.cpp
        RefTensorHandleFactory.hpp
    )

    add_subdirectory(workloads)

    if(BUILD_UNIT_TESTS)
        add_subdirectory(test)
    endif()

else()
    list(APPEND armnnRefBackend_sources
        RefBackendId.hpp
        RefLayerSupport.cpp
        RefLayerSupport.hpp
    )
endif()

add_library(armnnRefBackend OBJECT ${armnnRefBackend_sources})
target_include_directories(armnnRefBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnRefBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
target_include_directories(armnnRefBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
target_include_directories(armnnRefBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/profiling)
target_include_directories(armnnRefBackend PRIVATE ${PROJECT_SOURCE_DIR}/profiling/common/include)
target_include_directories(armnnRefBackend PRIVATE ${PROJECT_SOURCE_DIR}/profiling/client/include)
