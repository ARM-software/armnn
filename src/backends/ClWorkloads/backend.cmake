#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

if(ARMCOMPUTECL)
    add_subdirectory(${PROJECT_SOURCE_DIR}/src/backends/ClWorkloads)
    list(APPEND armnnLibraries armnnClBackend)
endif()
