#
# Copyright © 2017, 2023 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

if((NOT BUILD_BARE_METAL) AND (NOT EXECUTE_NETWORK_STATIC))

# File needed to wrap the existing backend into a dynamic one
list(APPEND armnnRefDynamicBackend_sources
    RefDynamicBackend.cpp
    RefDynamicBackend.hpp
)

# Set the backend source path
set(RefBackendPath ${PROJECT_SOURCE_DIR}/src/backends/reference)

# Source files of the backend, taken directly from the source tree
file(GLOB RefBackendBaseFiles ${RefBackendPath}/*.cpp)
file(GLOB RefBackendWorloadFiles ${RefBackendPath}/workloads/*.cpp)
set(RefBackendFiles ${RefBackendBaseFiles} ${RefBackendWorloadFiles})

# Remove the file that contains the static backend registration
list(REMOVE_ITEM RefBackendFiles ${RefBackendPath}/RefRegistryInitializer.cpp)

# Create the shared object
add_library(Arm_CpuRef_backend MODULE ${armnnRefDynamicBackend_sources} ${RefBackendFiles})
target_include_directories(Arm_CpuRef_backend PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(Arm_CpuRef_backend PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
target_include_directories(Arm_CpuRef_backend PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
target_include_directories(Arm_CpuRef_backend PRIVATE ${PROJECT_SOURCE_DIR}/src/profiling)
target_include_directories(Arm_CpuRef_backend PRIVATE ${PROJECT_SOURCE_DIR}/profiling/common/include)
target_include_directories(Arm_CpuRef_backend PRIVATE ${PROJECT_SOURCE_DIR}/profiling/client/include)
set_target_properties(Arm_CpuRef_backend PROPERTIES PREFIX "")
target_link_libraries(Arm_CpuRef_backend armnn)

# BUILD_BARE_METAL && EXECUTE_NETWORK_STATIC
endif()
