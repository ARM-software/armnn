#
# Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

# Read the ArmNN armnnTestUtils version components from file
file(READ ${CMAKE_CURRENT_LIST_DIR}/../include/armnnTestUtils/Version.hpp armnnTestUtilsVersion)

# Parse the ArmNN armnnTestUtils version components
string(REGEX MATCH "#define ARMNN_TEST_UTILS_MAJOR_VERSION ([0-9]*)" _ ${armnnTestUtilsVersion})
set(ARMNN_TEST_UTILS_MAJOR_VERSION ${CMAKE_MATCH_1})
string(REGEX MATCH "#define ARMNN_TEST_UTILS_MINOR_VERSION ([0-9]*)" _ ${armnnTestUtilsVersion})
set(ARMNN_TEST_UTILS_MINOR_VERSION ${CMAKE_MATCH_1})

# Define LIB version
set(ARMNN_TEST_UTILS_LIB_VERSION "${ARMNN_TEST_UTILS_MAJOR_VERSION}.${ARMNN_TEST_UTILS_MINOR_VERSION}")
# Define LIB soversion
set(ARMNN_TEST_UTILS_LIB_SOVERSION "${ARMNN_TEST_UTILS_MAJOR_VERSION}")