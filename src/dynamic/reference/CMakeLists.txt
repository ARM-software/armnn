#
# Copyright © 2019 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

cmake_minimum_required (VERSION 3.0.2)
project(dynamic-reference)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if("${CMAKE_SYSTEM_NAME}" STREQUAL Android)
    add_definitions("-DBOOST_ALL_NO_LIB") # Turn off auto-linking as we specify the libs manually
    find_package(Boost REQUIRED COMPONENTS system filesystem)
    include_directories(SYSTEM "${Boost_INCLUDE_DIRS}")
    link_directories(${Boost_LIBRARY_DIRS})
endif()

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

if("${CMAKE_SYSTEM_NAME}" STREQUAL Android)
    # Set the common backend source path
    set(CommonBackendPath ${PROJECT_SOURCE_DIR}/../../../src/backends/backendsCommon)
    # Source files of the common backend
    file(GLOB CommonBackendFiles ${CommonBackendPath}/*.cpp)

    set(ArmnnPath ${PROJECT_SOURCE_DIR}/../../../src/armnn)
    file(GLOB ArmnnFiles ${ArmnnPath}/*.cpp)

    file(GLOB LayerFiles ${ArmnnPath}/layers/*cpp)
    file(GLOB OptimizationFIles ${ArmnnPath}/optimizations/*.cpp)
    file(GLOB ArmnnBaseFiles ${ArmnnPath}/*cpp)

    set(ArmnnFiles ${ArmnnBaseFiles} ${LayerFiles} ${OptimizationFIles})

    set(ArmnnUtilsPath ${PROJECT_SOURCE_DIR}/../../../src/armnnUtils)
    file(GLOB ArmnnUtilsFiles ${ArmnnUtilsPath}/*cpp)

    set(ProfilingPath ${PROJECT_SOURCE_DIR}/../../../src/profiling)
    file(GLOB ProfilingFiles ${ProfilingPath}/*cpp)

    add_library(Arm_CpuRef_backend MODULE
        ${armnnRefDynamicBackend_sources} ${RefBackendFiles} ${CommonBackendFiles}
        ${ArmnnFiles} ${ArmnnUtilsFiles} ${ProfilingFiles})
else()
    # Create the shared object
    add_library(Arm_CpuRef_backend MODULE ${armnnRefDynamicBackend_sources} ${RefBackendFiles})
endif()

target_include_directories(Arm_CpuRef_backend PRIVATE ${PROJECT_SOURCE_DIR}/../../../include)
target_include_directories(Arm_CpuRef_backend PRIVATE ${PROJECT_SOURCE_DIR}/../../../third-party) # for half
target_include_directories(Arm_CpuRef_backend PRIVATE ${PROJECT_SOURCE_DIR}/../../../src/armnn)
target_include_directories(Arm_CpuRef_backend PRIVATE ${PROJECT_SOURCE_DIR}/../../../src/armnnUtils)
target_include_directories(Arm_CpuRef_backend PRIVATE ${PROJECT_SOURCE_DIR}/../../../src/backends)
target_include_directories(Arm_CpuRef_backend PRIVATE ${PROJECT_SOURCE_DIR}/../../../src/profiling)
if("${CMAKE_SYSTEM_NAME}" STREQUAL Android)
    target_include_directories(Arm_CpuRef_backend PRIVATE ${Boost_INCLUDE_DIRS})
    target_link_libraries(Arm_CpuRef_backend ${Boost_SYSTEM_LIBRARY} ${Boost_FILESYSTEM_LIBRARY})
endif()
