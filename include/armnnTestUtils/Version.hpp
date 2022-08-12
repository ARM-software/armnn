//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

/// Macro utils
#define STRINGIFY_VALUE(s) STRINGIFY_MACRO(s)
#define STRINGIFY_MACRO(s) #s

// ArmnnTestUtils version components
#define ARMNN_TEST_UTILS_MAJOR_VERSION 2
#define ARMNN_TEST_UTILS_MINOR_VERSION 0
#define ARMNN_TEST_UTILS_PATCH_VERSION 0

/// ARMNN_TEST_UTILS_VERSION: "X.Y.Z"
/// where:
///   X = Major version number
///   Y = Minor version number
///   Z = Patch version number
#define ARMNN_TEST_UTILS_VERSION STRINGIFY_VALUE(ARMNN_TEST_UTILS_MAJOR_VERSION) "." \
                                 STRINGIFY_VALUE(ARMNN_TEST_UTILS_MINOR_VERSION) "." \
                                 STRINGIFY_VALUE(ARMNN_TEST_UTILS_PATCH_VERSION)
