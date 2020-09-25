#
# Copyright © 2020 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

add_library(fmt INTERFACE)

target_compile_definitions(fmt INTERFACE FMT_HEADER_ONLY=1)

target_include_directories(fmt SYSTEM INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

install(
    TARGETS fmt
    EXPORT  armnn-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)