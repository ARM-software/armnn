#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

add_subdirectory(${PROJECT_SOURCE_DIR}/src/backends)
list(APPEND armnnLibraries armnnBackendsCommon)

# single place to use wildcards, so we can include
# yet unknown backend modules
FILE(GLOB backendIncludes ${PROJECT_SOURCE_DIR}/src/backends/*/backend.cmake)

foreach(backendInclude ${backendIncludes})
    include(${backendInclude})
endforeach()
