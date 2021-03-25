//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

/// Macro utils
#define STRINGIFY_VALUE(s) STRINGIFY_MACRO(s)
#define STRINGIFY_MACRO(s) #s

// ArmNN version components
#define ARMNN_MAJOR_VERSION 25
#define ARMNN_MINOR_VERSION 0
#define ARMNN_PATCH_VERSION 0

/// ARMNN_VERSION: "X.Y.Z"
/// where:
///   X = Major version number
///   Y = Minor version number
///   Z = Patch version number
#define ARMNN_VERSION STRINGIFY_VALUE(ARMNN_MAJOR_VERSION) "." \
                      STRINGIFY_VALUE(ARMNN_MINOR_VERSION) "." \
                      STRINGIFY_VALUE(ARMNN_PATCH_VERSION)
