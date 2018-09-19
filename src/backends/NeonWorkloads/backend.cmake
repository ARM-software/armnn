#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

if(ARMCOMPUTENEON)
    add_subdirectory(${PROJECT_SOURCE_DIR}/src/backends/NeonWorkloads)
    list(APPEND armnnLibraries armnnNeonBackend)
endif()
