#
# Copyright © 2018-2024 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

include(CheckIncludeFiles)

if (BUILD_ARMNN_SERIALIZER
        OR BUILD_TF_LITE_PARSER
        OR BUILD_ONNX_PARSER
        OR BUILD_CLASSIC_DELEGATE
        OR BUILD_OPAQUE_DELEGATE)
    set(ExecuteNetwork_sources
        ExecuteNetwork/IExecutor.hpp
        ExecuteNetwork/ArmNNExecutor.cpp
        ExecuteNetwork/ArmNNExecutor.hpp
        ExecuteNetwork/ExecuteNetwork.cpp
        ExecuteNetwork/ExecuteNetworkProgramOptions.cpp
        ExecuteNetwork/ExecuteNetworkProgramOptions.hpp
        ExecuteNetwork/ExecuteNetworkParams.cpp
        ExecuteNetwork/ExecuteNetworkParams.hpp
        ExecuteNetwork/FileComparisonExecutor.cpp
        ExecuteNetwork/FileComparisonExecutor.hpp
        NetworkExecutionUtils/NetworkExecutionUtils.cpp
        NetworkExecutionUtils/NetworkExecutionUtils.hpp)

    if(BUILD_CLASSIC_DELEGATE OR BUILD_OPAQUE_DELEGATE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-comment")
        set(ExecuteNetwork_sources
            ${ExecuteNetwork_sources}
            ExecuteNetwork/TfliteExecutor.cpp
            ExecuteNetwork/TfliteExecutor.hpp)
    endif()

    add_executable_ex(ExecuteNetwork ${ExecuteNetwork_sources})
    target_include_directories(ExecuteNetwork PRIVATE ../src/armnn
                                                      ../src/armnnUtils
                                                      ../src/backends
                                                      ./NetworkExecutionUtils
                                                      ${CMAKE_CURRENT_SOURCE_DIR})

    if(EXECUTE_NETWORK_STATIC)
        target_link_libraries(ExecuteNetwork
                -Wl,--whole-archive
                armnnSerializer
                armnnTfLiteParser
                armnn
                pthread
                -Wl,--no-whole-archive
                )
    else()
        if (BUILD_ARMNN_SERIALIZER)
            target_link_libraries(ExecuteNetwork armnnSerializer)
        endif()
        if (BUILD_TF_LITE_PARSER)
            target_link_libraries(ExecuteNetwork armnnTfLiteParser)
        endif()
        if (BUILD_ONNX_PARSER)
            target_link_libraries(ExecuteNetwork armnnOnnxParser)
        endif()
        if (BUILD_CLASSIC_DELEGATE)
            target_link_libraries(ExecuteNetwork ArmnnDelegate::ArmnnDelegate)
        endif()
        if (BUILD_OPAQUE_DELEGATE)
            target_link_libraries(ExecuteNetwork ArmnnDelegate::ArmnnOpaqueDelegate)
        endif()
        target_link_libraries(ExecuteNetwork armnn)
    endif()

   target_link_libraries(ExecuteNetwork ${CMAKE_THREAD_LIBS_INIT})
   addDllCopyCommands(ExecuteNetwork)
endif()
