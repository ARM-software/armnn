#
# Copyright © 2019 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

cmake_minimum_required (VERSION 3.0.2)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# File needed to wrap the existing backend into a dynamic one
list(APPEND armnnRefDynamicBackend_sources
    RefDynamicBackend.cpp
    RefDynamicBackend.hpp
)

# Set the backend source path
set(RefBackendPath ${PROJECT_SOURCE_DIR}/../../../src/backends/reference)

# Source files of the backend, taken directly from the source tree
file(GLOB RefBackendBaseFiles ${RefBackendPath}/*.cpp)
file(GLOB RefBackendWorloadFiles ${RefBackendPath}/workloads/*.cpp)
set(RefBackendFiles ${RefBackendBaseFiles} ${RefBackendWorloadFiles})

# Remove the file that contains the static backend registration
list(REMOVE_ITEM RefBackendFiles ${RefBackendPath}/RefRegistryInitializer.cpp)

# Create the shared object
add_library(Arm_CpuRef_backend MODULE ${armnnRefDynamicBackend_sources} ${RefBackendFiles})

target_include_directories(Arm_CpuRef_backend PRIVATE ${PROJECT_SOURCE_DIR}/../../../include)
target_include_directories(Arm_CpuRef_backend PRIVATE ${PROJECT_SOURCE_DIR}/../../../third-party) # for half
target_include_directories(Arm_CpuRef_backend PRIVATE ${PROJECT_SOURCE_DIR}/../../../src/armnn)
target_include_directories(Arm_CpuRef_backend PRIVATE ${PROJECT_SOURCE_DIR}/../../../src/armnnUtils)
target_include_directories(Arm_CpuRef_backend PRIVATE ${PROJECT_SOURCE_DIR}/../../../src/backends)
target_include_directories(Arm_CpuRef_backend PRIVATE ${PROJECT_SOURCE_DIR}/../../../src/profiling)