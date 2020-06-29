#
# Copyright © 2020 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

if(BUILD_TIMELINE_DECODER)
    set(timelineDecoder_sources)
    list(APPEND timelineDecoder_sources
        ${PROJECT_SOURCE_DIR}/include/armnn/profiling/ITimelineDecoder.hpp
        TimelineCaptureCommandHandler.cpp
        TimelineCaptureCommandHandler.hpp
        JSONTimelineDecoder.cpp
        JSONTimelineDecoder.hpp
        TimelineDecoder.cpp
        TimelineDecoder.hpp
        TimelineDirectoryCaptureCommandHandler.cpp
        TimelineDirectoryCaptureCommandHandler.hpp)

    include_directories(${PROJECT_SOURCE_DIR}/src/profiling
                        ${PROJECT_SOURCE_DIR}/profiling/common/include
                        ${PROJECT_SOURCE_DIR}/src/armnnUtils)

    if(BUILD_UNIT_TESTS)
      target_include_directories(UnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/timelineDecoder)
    endif()

    add_library_ex(timelineDecoder SHARED ${timelineDecoder_sources})

    set_target_properties(timelineDecoder PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
    set_target_properties(timelineDecoder PROPERTIES VERSION ${GENERIC_LIB_VERSION} SOVERSION ${GENERIC_LIB_SOVERSION} )

    target_link_libraries(timelineDecoder armnn)

    install(TARGETS timelineDecoder
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()


