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
            ConnectionHandler.hpp
            )

    include_directories(src/armnnUtils src/profiling)

    add_library_ex(armnnBasePipeServer SHARED ${BasePipeServer_sources})
    set_target_properties(armnnBasePipeServer PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
    set_target_properties(armnnBasePipeServer PROPERTIES VERSION ${GENERIC_LIB_VERSION}
                          SOVERSION ${GENERIC_LIB_SOVERSION})

    install(TARGETS armnnBasePipeServer
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()