#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

list(APPEND armnnClBackend_sources
    ClBackend.cpp
    ClBackend.hpp
    ClContextControl.cpp
    ClContextControl.hpp
    ClLayerSupport.cpp
    ClLayerSupport.hpp
    ClWorkloadFactory.cpp
    ClWorkloadFactory.hpp
)

if(ARMCOMPUTECL)
    list(APPEND armnnClBackend_sources
        OpenClTimer.cpp
        OpenClTimer.hpp
    )

    add_subdirectory(workloads)
    add_subdirectory(test)
endif()

add_library(armnnClBackend STATIC ${armnnClBackend_sources})
target_include_directories(armnnClBackend PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(armnnClBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnClBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
