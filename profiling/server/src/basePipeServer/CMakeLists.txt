#
# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

if(BUILD_BASE_PIPE_SERVER)
    set(BasePipeServer_sources)
    list(APPEND BasePipeServer_sources
        BasePipeServer.cpp)

     if(NOT "${TOOLCHAIN_PREFIX}" STREQUAL x86_64-w64-mingw32)
         list(APPEND BasePipeServer_sources
                 ConnectionHandler.cpp)
     endif()

    include_directories(${PROJECT_SOURCE_DIR}/profiling/common/include)
    include_directories(${PROJECT_SOURCE_DIR}/profiling/server/include/basePipeServer)

    if (BUILD_UNIT_TESTS)
        target_include_directories(UnitTests PRIVATE ${PROJECT_SOURCE_DIR}/profiling/server/src/basePipeServer)
        target_include_directories(UnitTests PUBLIC ${PROJECT_SOURCE_DIR}/profiling/common/include)
    endif()

    if (BUILD_STATIC_PIPE_LIBS)
      add_library_ex(armnnBasePipeServer STATIC ${BasePipeServer_sources})
      target_link_libraries(armnnBasePipeServer pipeCommon)

      if ("${CMAKE_SYSTEM_NAME}" STREQUAL Windows)
        target_link_libraries(armnnBasePipeServer ws2_32.lib)
      endif()
    else()
       add_library_ex(armnnBasePipeServer SHARED ${BasePipeServer_sources})
       target_link_libraries(armnnBasePipeServer pipeCommon)

       if ("${CMAKE_SYSTEM_NAME}" STREQUAL Windows)
         target_link_libraries(armnnBasePipeServer ws2_32.lib)
       endif()

       set_target_properties(armnnBasePipeServer PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
       set_target_properties(armnnBasePipeServer PROPERTIES VERSION ${GENERIC_LIB_VERSION}
                                                            SOVERSION ${GENERIC_LIB_SOVERSION})

       target_include_directories(armnnBasePipeServer PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)

       install(TARGETS armnnBasePipeServer
               LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
               RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
     endif()
endif()
