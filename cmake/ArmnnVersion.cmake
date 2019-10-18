#
# Copyright © 2019 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

# Read the ArmNN version components from file
file(READ ${CMAKE_CURRENT_LIST_DIR}/../include/armnn/Version.hpp armnnVersion)

# Parse the ArmNN version components
string(REGEX MATCH "#define ARMNN_MAJOR_VERSION ([0-9]*)" _ ${armnnVersion})
set(ARMNN_MAJOR_VERSION ${CMAKE_MATCH_1})
string(REGEX MATCH "#define ARMNN_MINOR_VERSION ([0-9]*)" _ ${armnnVersion})
set(ARMNN_MINOR_VERSION ${CMAKE_MATCH_1})

# Define LIB version
set(GENERIC_LIB_VERSION "${ARMNN_MAJOR_VERSION}.${ARMNN_MINOR_VERSION}")

# Define LIB soversion
set(GENERIC_LIB_SOVERSION "${ARMNN_MAJOR_VERSION}")
