#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

if(ARMCOMPUTENEON)
    list(APPEND armnnNeonBackend_sources
        NeonBackend.cpp
        NeonBackend.hpp
        NeonBackendId.hpp
        NeonBackendModelContext.hpp
        NeonBackendModelContext.cpp
        NeonInterceptorScheduler.hpp
        NeonInterceptorScheduler.cpp
        NeonLayerSupport.cpp
        NeonLayerSupport.hpp
        NeonRegistryInitializer.cpp
        NeonTensorHandle.hpp
        NeonTensorHandleFactory.cpp
        NeonTensorHandleFactory.hpp
        NeonTimer.hpp
        NeonTimer.cpp
        NeonWorkloadFactory.cpp
        NeonWorkloadFactory.hpp
    )

    add_subdirectory(workloads)

    if(BUILD_UNIT_TESTS)
        add_subdirectory(test)
    endif()

else()
    list(APPEND armnnNeonBackend_sources
        NeonBackendId.hpp
        NeonLayerSupport.cpp
        NeonLayerSupport.hpp
    )
endif()

add_library(armnnNeonBackend OBJECT ${armnnNeonBackend_sources})
target_include_directories(armnnNeonBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnNeonBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
target_include_directories(armnnNeonBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
target_include_directories(armnnNeonBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/profiling)
target_include_directories(armnnNeonBackend PRIVATE ${PROJECT_SOURCE_DIR}/profiling/common/include)
