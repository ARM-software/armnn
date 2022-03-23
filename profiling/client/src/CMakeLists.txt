#
# Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

    set(pipeClient_sources)
    list(APPEND pipeClient_sources
        ActivateTimelineReportingCommandHandler.cpp
        ActivateTimelineReportingCommandHandler.hpp
        backends/BackendProfiling.cpp
        backends/BackendProfiling.hpp
        backends/IBackendProfiling.cpp
        BufferManager.cpp
        BufferManager.hpp
        CommandHandler.cpp
        CommandHandler.hpp
        ConnectionAcknowledgedCommandHandler.cpp
        ConnectionAcknowledgedCommandHandler.hpp
        CounterIdMap.cpp
        DeactivateTimelineReportingCommandHandler.cpp
        DeactivateTimelineReportingCommandHandler.hpp
        FileOnlyProfilingConnection.cpp
        FileOnlyProfilingConnection.hpp
        Holder.cpp
        IBufferManager.hpp
        IConsumer.hpp
        INotifyBackends.hpp
        IPacketBuffer.hpp
        IPeriodicCounterCapture.hpp
        IProfilingConnectionFactory.hpp
        IProfilingConnection.hpp
        IProfilingService.cpp
        ISendThread.hpp
        NullProfilingConnection.hpp
        PacketBuffer.cpp
        PacketBuffer.hpp
        PeriodicCounterCapture.cpp
        PeriodicCounterCapture.hpp
        PeriodicCounterSelectionCommandHandler.cpp
        PeriodicCounterSelectionCommandHandler.hpp
        PerJobCounterSelectionCommandHandler.cpp
        PerJobCounterSelectionCommandHandler.hpp
        ProfilingConnectionDumpToFileDecorator.cpp
        ProfilingConnectionDumpToFileDecorator.hpp
        ProfilingConnectionFactory.cpp
        ProfilingConnectionFactory.hpp
        ProfilingService.cpp
        ProfilingService.hpp
        ProfilingStateMachine.cpp
        ProfilingStateMachine.hpp
        ProfilingUtils.cpp
        ProfilingUtils.hpp
        RegisterBackendCounters.cpp
        RegisterBackendCounters.hpp
        RequestCounterDirectoryCommandHandler.cpp
        RequestCounterDirectoryCommandHandler.hpp
        SendCounterPacket.cpp
        SendCounterPacket.hpp
        SendThread.cpp
        SendThread.hpp
        SendTimelinePacket.cpp
        SendTimelinePacket.hpp
        SocketProfilingConnection.cpp
        SocketProfilingConnection.hpp
        TimelinePacketWriterFactory.cpp
        TimelinePacketWriterFactory.hpp
        TimelineUtilityMethods.cpp)

    include_directories(${PROJECT_SOURCE_DIR}/profiling/common/include)
    include_directories(${PROJECT_SOURCE_DIR}/common/include)
    include_directories(${PROJECT_SOURCE_DIR}/profiling/client/include)
    include_directories(${PROJECT_SOURCE_DIR}/client/include)

    # for the moment will only build a static version of this common code
    # to simplify the build. No extra .so file to deploy to boards etc.
    add_library_ex(pipeClient STATIC ${pipeClient_sources})

    target_link_libraries(pipeClient pipeCommon)

    target_compile_definitions(pipeClient PRIVATE "ARMNN_COMPILING_DLL")

    # install the target
    install(
        TARGETS pipeClient
        EXPORT  armnn-targets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
