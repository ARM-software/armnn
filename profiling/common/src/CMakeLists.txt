#
# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

    set(pipeCommon_sources)
    list(APPEND pipeCommon_sources
        CommandHandlerFunctor.cpp
        CommandHandlerKey.cpp
        CommandHandlerRegistry.cpp
        CommonProfilingUtils.cpp
        CounterDirectory.cpp
        LabelsAndEventClasses.cpp
        Logging.cpp
        NetworkSockets.cpp
        PacketVersionResolver.cpp
        Processes.cpp
        SwTrace.cpp
        Threads.cpp)

    include_directories(${PROJECT_SOURCE_DIR}/profiling/common/include)
    include_directories(${PROJECT_SOURCE_DIR}/common/include)

    # will only build a static version of this common code
    # to simplify the build. No extra .so file to deploy to boards etc.
    add_library_ex(pipeCommon STATIC ${pipeCommon_sources})
    target_link_libraries(pipeCommon fmt)

    target_compile_definitions(pipeCommon PRIVATE "ARMNN_COMPILING_DLL")

    # install the target
    install(
        TARGETS pipeCommon
        EXPORT  armnn-targets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
