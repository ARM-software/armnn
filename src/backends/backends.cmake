#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

add_subdirectory(${PROJECT_SOURCE_DIR}/src/backends)
list(APPEND armnnLibraries armnnBackendsCommon)

# single place to use wildcards, so we can include
# yet unknown backend modules and corresponding common libraries
FILE(GLOB commonIncludes ${PROJECT_SOURCE_DIR}/src/backends/*/common.cmake)
FILE(GLOB backendIncludes ${PROJECT_SOURCE_DIR}/src/backends/*/backend.cmake)

# prefer to include common code first so backends can depend on them
foreach(includeFile ${commonIncludes} ${backendIncludes})
    message("Including backend into the build: ${includeFile}")
    include(${includeFile})
endforeach()
