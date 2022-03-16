#
# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

if(BUILD_TIMELINE_DECODER)
    set(timelineDecoder_sources)
    list(APPEND timelineDecoder_sources
        DirectoryCaptureCommandHandler.cpp
        TimelineCaptureCommandHandler.cpp
        TimelineDecoder.cpp
        TimelineDirectoryCaptureCommandHandler.cpp)

    include_directories(${PROJECT_SOURCE_DIR}/profiling/common/include)

    if(BUILD_UNIT_TESTS)
        include_directories(${PROJECT_SOURCE_DIR}/src/profiling
                            ${PROJECT_SOURCE_DIR}/src/armnnUtils)
        target_include_directories(UnitTests PRIVATE ${PROJECT_SOURCE_DIR}/profiling/server/include/timelineDecoder)
    endif()


    if (BUILD_STATIC_PIPE_LIBS)
      add_library_ex(timelineDecoder STATIC ${timelineDecoder_sources})
      target_link_libraries(timelineDecoder pipeCommon)

      if ("${CMAKE_SYSTEM_NAME}" STREQUAL Windows)
        target_link_libraries(timelineDecoder ws2_32.lib)
      endif()
    else()
      add_library_ex(timelineDecoder SHARED ${timelineDecoder_sources})
      target_link_libraries(timelineDecoder pipeCommon)

      if ("${CMAKE_SYSTEM_NAME}" STREQUAL Windows)
        target_link_libraries(timelineDecoder ws2_32.lib)
      endif()

      set_target_properties(timelineDecoder PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
      set_target_properties(timelineDecoder PROPERTIES VERSION ${GENERIC_LIB_VERSION} SOVERSION ${GENERIC_LIB_SOVERSION} )

      install(TARGETS timelineDecoder
              LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
              RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    endif()
endif()
