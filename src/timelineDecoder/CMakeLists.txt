#
# Copyright © 2020 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

set(timelineDecoder_sources)
list(APPEND timelineDecoder_sources
        ../../include/armnn/profiling/ITimelineDecoder.hpp
        TimelineCaptureCommandHandler.cpp
        TimelineCaptureCommandHandler.hpp
        TimelineDecoder.cpp
        TimelineDecoder.hpp
        TimelineDirectoryCaptureCommandHandler.cpp
        TimelineDirectoryCaptureCommandHandler.hpp
        )

include_directories(../timelineDecoder ../profiling)

add_library_ex(timelineDecoder SHARED ${timelineDecoder_sources})

set_target_properties(timelineDecoder PROPERTIES VERSION ${GENERIC_LIB_VERSION} SOVERSION ${GENERIC_LIB_SOVERSION} )