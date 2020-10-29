#
# Copyright © 2020 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

add_library(fmt STATIC src/format.cc)
target_include_directories(fmt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

install(
    TARGETS fmt
    EXPORT  armnn-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)