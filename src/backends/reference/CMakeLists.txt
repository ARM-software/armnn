#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

list(APPEND armnnRefBackend_sources
    RefLayerSupport.cpp
    RefLayerSupport.hpp
    RefWorkloadFactory.cpp
    RefWorkloadFactory.hpp
)

add_library(armnnRefBackend STATIC ${armnnRefBackend_sources})
target_include_directories(armnnRefBackend PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(armnnRefBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnRefBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)

add_subdirectory(workloads test)

