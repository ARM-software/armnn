#
# Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

# Read the ArmNN Delegate version components from file
file(READ ${CMAKE_CURRENT_LIST_DIR}/../delegate/opaque/include/Version.hpp opaqueDelegateVersion)

# Parse the ArmNN Delegate version components
string(REGEX MATCH "#define OPAQUE_DELEGATE_MAJOR_VERSION ([0-9]*)" _ ${opaqueDelegateVersion})
set(OPAQUE_DELEGATE_MAJOR_VERSION ${CMAKE_MATCH_1})
string(REGEX MATCH "#define OPAQUE_DELEGATE_MINOR_VERSION ([0-9]*)" _ ${opaqueDelegateVersion})
set(OPAQUE_DELEGATE_MINOR_VERSION ${CMAKE_MATCH_1})

# Define LIB version
set(OPAQUE_DELEGATE_LIB_VERSION "${OPAQUE_DELEGATE_MAJOR_VERSION}.${OPAQUE_DELEGATE_MINOR_VERSION}")
# Define LIB soversion
set(OPAQUE_DELEGATE_LIB_SOVERSION "${OPAQUE_DELEGATE_MINOR_VERSION}")