#
# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

cmake_minimum_required (VERSION 3.0.2)
project(sample-dynamic)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND armnnSampleDynamicBackend_sources
        SampleDynamicAdditionWorkload.cpp
        SampleDynamicAdditionWorkload.hpp
        SampleDynamicBackend.cpp
        SampleDynamicBackend.hpp
        SampleDynamicLayerSupport.cpp
        SampleDynamicLayerSupport.hpp
        SampleDynamicWorkloadFactory.cpp
        SampleDynamicWorkloadFactory.hpp
        SampleMemoryManager.cpp
        SampleMemoryManager.hpp
        SampleTensorHandle.cpp
        SampleTensorHandle.hpp
        SampleDynamicTensorHandleFactory.cpp
        SampleDynamicTensorHandleFactory.hpp
)

add_library(Arm_SampleDynamic_backend MODULE ${armnnSampleDynamicBackend_sources})

target_include_directories(Arm_SampleDynamic_backend PRIVATE ${PROJECT_SOURCE_DIR}/../../../include)
target_include_directories(Arm_SampleDynamic_backend PRIVATE ${PROJECT_SOURCE_DIR}/../../../third-party)
target_include_directories(Arm_SampleDynamic_backend PRIVATE ${PROJECT_SOURCE_DIR}/../../../src/armnn)
target_include_directories(Arm_SampleDynamic_backend PRIVATE ${PROJECT_SOURCE_DIR}/../../../src/armnnUtils)
target_include_directories(Arm_SampleDynamic_backend PRIVATE ${PROJECT_SOURCE_DIR}/../../../src/backends)
target_include_directories(Arm_SampleDynamic_backend PRIVATE ${PROJECT_SOURCE_DIR}/../../../src/profiling)
target_include_directories(Arm_SampleDynamic_backend PRIVATE ${PROJECT_SOURCE_DIR}/../../../profiling/common/include)
target_include_directories(Arm_SampleDynamic_backend PRIVATE ${PROJECT_SOURCE_DIR}/../../../profiling)
target_link_libraries(Arm_SampleDynamic_backend ${ARMNN_PATH})
