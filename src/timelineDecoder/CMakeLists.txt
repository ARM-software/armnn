#
# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

if(BUILD_TIMELINE_DECODER)
    set(jsonTimelineDecoder_sources)
    list(APPEND jsonTimelineDecoder_sources
        JSONTimelineDecoder.cpp
        JSONTimelineDecoder.hpp)

    include_directories(${PROJECT_SOURCE_DIR}/src/profiling
                        ${PROJECT_SOURCE_DIR}/profiling/common/include
                        ${PROJECT_SOURCE_DIR}/profiling/server/include/timelineDecoder
                        ${PROJECT_SOURCE_DIR}/src/armnnUtils)

    if(BUILD_UNIT_TESTS)
      target_include_directories(UnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/timelineDecoder)
    endif()

    add_library_ex(jsonTimelineDecoder SHARED ${jsonTimelineDecoder_sources})

    set_target_properties(jsonTimelineDecoder PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
    set_target_properties(jsonTimelineDecoder PROPERTIES VERSION ${GENERIC_LIB_VERSION} SOVERSION ${GENERIC_LIB_SOVERSION} )

    target_link_libraries(jsonTimelineDecoder armnn)

    install(TARGETS jsonTimelineDecoder
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()


