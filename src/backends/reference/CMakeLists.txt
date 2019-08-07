#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

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
    RefTensorHandleFactory.hpp
    RefTensorHandleFactory.cpp
)

add_library(armnnRefBackend OBJECT ${armnnRefBackend_sources})
target_include_directories(armnnRefBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnRefBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
target_include_directories(armnnRefBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)

add_subdirectory(workloads)

if(BUILD_UNIT_TESTS)
    add_subdirectory(test)
endif()
