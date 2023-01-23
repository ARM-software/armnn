#
# Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

if(ARMNNTOSAREF)
    list(APPEND armnnTosaRefBackend_sources
        TosaRefBackend.cpp
        TosaRefBackend.hpp
        TosaRefBackendId.hpp
        TosaRefTensorHandle.hpp
        TosaRefTensorHandle.cpp
        TosaRefLayerSupport.cpp
        TosaRefLayerSupport.hpp
        TosaRefMemoryManager.hpp
        TosaRefMemoryManager.cpp
        TosaRefRegistryInitializer.cpp
        TosaRefWorkloadFactory.cpp
        TosaRefWorkloadFactory.hpp
        TosaRefTensorHandleFactory.cpp
        TosaRefTensorHandleFactory.hpp
    )

    add_subdirectory(workloads)

    if(BUILD_UNIT_TESTS)
        add_subdirectory(test)
    endif()

else()
    list(APPEND armnnTosaRefBackend_sources
        TosaRefBackendId.hpp
    )
endif()

add_library(armnnTosaRefBackend OBJECT ${armnnTosaRefBackend_sources})
set_target_properties(armnnTosaRefBackend PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(armnnTosaRefBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnTosaRefBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
target_include_directories(armnnTosaRefBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
target_include_directories(armnnTosaRefBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/backends/tosaCommon)
