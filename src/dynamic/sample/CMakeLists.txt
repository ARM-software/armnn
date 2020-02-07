#
# Copyright © 2020 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

cmake_minimum_required (VERSION 3.0.2)
project(sample-dynamic)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_definitions("-DBOOST_ALL_NO_LIB") # Turn off auto-linking as we specify the libs manually
find_package(Boost REQUIRED COMPONENTS system filesystem)
include_directories(SYSTEM "${Boost_INCLUDE_DIRS}")
link_directories(${Boost_LIBRARY_DIRS})

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
)

add_library(Arm_SampleDynamic_backend MODULE ${armnnSampleDynamicBackend_sources})

target_include_directories(Arm_SampleDynamic_backend PRIVATE ${PROJECT_SOURCE_DIR}/../../../include)
target_include_directories(Arm_SampleDynamic_backend PRIVATE ${PROJECT_SOURCE_DIR}/../../../third-party)
target_include_directories(Arm_SampleDynamic_backend PRIVATE ${PROJECT_SOURCE_DIR}/../../../src/armnn)
target_include_directories(Arm_SampleDynamic_backend PRIVATE ${PROJECT_SOURCE_DIR}/../../../src/armnnUtils)
target_include_directories(Arm_SampleDynamic_backend PRIVATE ${PROJECT_SOURCE_DIR}/../../../src/backends)
target_include_directories(Arm_SampleDynamic_backend PRIVATE ${PROJECT_SOURCE_DIR}/../../../src/profiling)
target_include_directories(Arm_SampleDynamic_backend PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(Arm_SampleDynamic_backend ${Boost_SYSTEM_LIBRARY} ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries(Arm_SampleDynamic_backend ${ARMNN_PATH})
