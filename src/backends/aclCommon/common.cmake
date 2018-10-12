#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

if(ARMCOMPUTENEON OR ARMCOMPUTECL)
    add_subdirectory(${PROJECT_SOURCE_DIR}/src/backends/aclCommon)
    list(APPEND armnnLibraries armnnAclCommon)
    list(APPEND armnnUnitTestLibraries armnnAclCommonUnitTests)
endif()
