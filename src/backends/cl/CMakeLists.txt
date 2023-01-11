#
# Copyright © 2017-2023 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

if(ARMCOMPUTECL)
    find_program(FLATC flatc
                 HINTS ${FLATC_DIR}
                 DOC "Path to 'flatc', the flatbuffers compiler")
    if (NOT FLATC)
        message(WARNING "flatc not found. Specify the full path of the flatc executable with -DFLATC=<flatc path>")
    else()
        add_custom_command(
            # Updates ClContextSchema_generated.h if changes are made.
            OUTPUT ClContextSchema_generated.h DEPENDS ClContextSchema.fbs
            COMMAND ${FLATC} -o ${CMAKE_CURRENT_BINARY_DIR} --cpp ${CMAKE_CURRENT_SOURCE_DIR}/ClContextSchema.fbs
        )
    endif()

    list(APPEND armnnClBackend_sources
        ClContextSchema_generated.h
        ClBackend.cpp
        ClBackend.hpp
        ClBackendContext.cpp
        ClBackendContext.hpp
        ClBackendDefaultAllocator.hpp
        ClBackendId.hpp
        ClBackendModelContext.cpp
        ClBackendModelContext.hpp
        ClContextControl.cpp
        ClContextControl.hpp
        ClContextDeserializer.hpp
        ClContextDeserializer.cpp
        ClContextSerializer.hpp
        ClContextSerializer.cpp
        ClImportTensorHandle.hpp
        ClImportTensorHandleFactory.cpp
        ClImportTensorHandleFactory.hpp
        ClLayerSupport.cpp
        ClLayerSupport.hpp
        ClRegistryInitializer.cpp
        ClTensorHandle.hpp
        ClTensorHandleFactory.cpp
        ClTensorHandleFactory.hpp
        ClWorkloadFactory.cpp
        ClWorkloadFactory.hpp
        ICLTensorProxy.hpp
        OpenClTimer.cpp
        OpenClTimer.hpp
    )

    add_subdirectory(workloads)

    if(BUILD_UNIT_TESTS)
        add_subdirectory(test)
    endif()

    include_directories(SYSTEM "${FLATBUFFERS_INCLUDE_PATH}")

else()
    list(APPEND armnnClBackend_sources
        ClBackendId.hpp
        ClLayerSupport.cpp
        ClLayerSupport.hpp
    )
endif()

add_library(armnnClBackend OBJECT ${armnnClBackend_sources})
target_include_directories(armnnClBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnClBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
target_include_directories(armnnClBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
target_include_directories(armnnClBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/profiling)
target_include_directories(armnnClBackend PRIVATE ${PROJECT_SOURCE_DIR}/profiling/common/include)
target_include_directories(armnnClBackend PRIVATE ${PROJECT_SOURCE_DIR}/profiling/client/include)
