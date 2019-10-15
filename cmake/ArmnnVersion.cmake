#
# Copyright © 2019 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

# Read the ArmNN version components from file
file(READ ${CMAKE_CURRENT_LIST_DIR}/../ArmnnVersion.txt armnnVersion)

# Parse the ArmNN version components
string(REGEX MATCH "ARMNN_MAJOR_VERSION ([0-9]*)" _ ${armnnVersion})
set(ARMNN_MAJOR_VERSION ${CMAKE_MATCH_1})
string(REGEX MATCH "ARMNN_MINOR_VERSION ([0-9]*)" _ ${armnnVersion})
set(ARMNN_MINOR_VERSION ${CMAKE_MATCH_1})
string(REGEX MATCH "ARMNN_PATCH_VERSION ([0-9]*)" _ ${armnnVersion})
set(ARMNN_PATCH_VERSION ${CMAKE_MATCH_1})

# Put together the ArmNN version (YYYYMMPP)
set(ARMNN_VERSION "20${ARMNN_MAJOR_VERSION}${ARMNN_MINOR_VERSION}${ARMNN_PATCH_VERSION}")

# Pass the ArmNN version to the build
if(${CMAKE_VERSION} VERSION_LESS "3.12.0")
    add_definitions(-DARMNN_VERSION_FROM_FILE=${ARMNN_VERSION})
else()
    add_compile_definitions(ARMNN_VERSION_FROM_FILE=${ARMNN_VERSION})
endif()
