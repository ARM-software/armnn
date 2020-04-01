#
# Copyright © 2020 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

if(BUILD_TIMELINE_DECODER)
    set(timelineDecoder_sources)
    list(APPEND timelineDecoder_sources
            ../../include/armnn/profiling/ITimelineDecoder.hpp
            TimelineCaptureCommandHandler.cpp
            TimelineCaptureCommandHandler.hpp
            JSONTimelineDecoder.cpp
            JSONTimelineDecoder.hpp
            TimelineDecoder.cpp
            TimelineDecoder.hpp
            TimelineDirectoryCaptureCommandHandler.cpp
            TimelineDirectoryCaptureCommandHandler.hpp
            )

    include_directories(../timelineDecoder ../profiling)

    add_library_ex(timelineDecoder SHARED ${timelineDecoder_sources})
    set_target_properties(timelineDecoder PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
    set_target_properties(timelineDecoder PROPERTIES VERSION ${GENERIC_LIB_VERSION} SOVERSION ${GENERIC_LIB_SOVERSION} )

    install(TARGETS timelineDecoder
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()


