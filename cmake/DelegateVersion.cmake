#
# Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

# Read the ArmNN Delegate version components from file
file(READ ${CMAKE_CURRENT_LIST_DIR}/../delegate/include/Version.hpp delegateVersion)

# Parse the ArmNN Delegate version components
string(REGEX MATCH "#define DELEGATE_MAJOR_VERSION ([0-9]*)" _ ${delegateVersion})
set(DELEGATE_MAJOR_VERSION ${CMAKE_MATCH_1})
string(REGEX MATCH "#define DELEGATE_MINOR_VERSION ([0-9]*)" _ ${delegateVersion})
set(DELEGATE_MINOR_VERSION ${CMAKE_MATCH_1})

# Define LIB version
set(DELEGATE_LIB_VERSION "${DELEGATE_MAJOR_VERSION}.${DELEGATE_MINOR_VERSION}")
# Define LIB soversion
set(DELEGATE_LIB_SOVERSION "${DELEGATE_MAJOR_VERSION}")