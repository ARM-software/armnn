#
# Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

list(APPEND armnnTosaRefBackendWorkloads_sources
        TosaRefBaseWorkload.hpp
        TosaRefPreCompiledWorkload.cpp
        TosaRefPreCompiledWorkload.hpp
)

add_library(armnnTosaRefBackendWorkloads OBJECT ${armnnTosaRefBackendWorkloads_sources})
target_include_directories(armnnTosaRefBackendWorkloads PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnTosaRefBackendWorkloads PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
target_include_directories(armnnTosaRefBackendWorkloads PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
target_include_directories(armnnTosaRefBackendWorkloads PRIVATE ${PROJECT_SOURCE_DIR}/src/profiling)
target_include_directories(armnnTosaRefBackendWorkloads PRIVATE ${PROJECT_SOURCE_DIR}/profiling/common/include)
target_include_directories(armnnTosaRefBackendWorkloads PRIVATE ${PROJECT_SOURCE_DIR}/profiling/client/include)
