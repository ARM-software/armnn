#
# Copyright © 2020 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

if(BUILD_BASE_PIPE_SERVER)
    set(BasePipeServer_sources)
    list(APPEND BasePipeServer_sources
        BasePipeServer.cpp
        BasePipeServer.hpp
        ConnectionHandler.cpp
        ConnectionHandler.hpp)

    include_directories(${PROJECT_SOURCE_DIR}/profiling/common/include)

    if (BUILD_UNIT_TESTS)
        target_include_directories(UnitTests PRIVATE ${PROJECT_SOURCE_DIR}/profiling/server/src/basePipeServer)
        target_include_directories(UnitTests PUBLIC ${PROJECT_SOURCE_DIR}/profiling/common/include)
    endif()

    add_library_ex(armnnBasePipeServer SHARED ${BasePipeServer_sources})

    set_target_properties(armnnBasePipeServer PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
    set_target_properties(armnnBasePipeServer PROPERTIES VERSION ${GENERIC_LIB_VERSION}
                          SOVERSION ${GENERIC_LIB_SOVERSION})

    target_link_libraries(armnnBasePipeServer armnn)
    if ("${CMAKE_SYSTEM_NAME}" STREQUAL Windows)
        target_link_libraries(armnnBasePipeServer Ws2_32.lib)
    endif()

    install(TARGETS armnnBasePipeServer
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()