//
// Copyright © 2023-2025 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

namespace armnnDelegate
{

/// Macro utils
#define STRINGIFY_VALUE(s) STRINGIFY_MACRO(s)
#define STRINGIFY_MACRO(s) #s

// ArmNN Delegate version components
#define DELEGATE_MAJOR_VERSION 29
#define DELEGATE_MINOR_VERSION 1
#define DELEGATE_PATCH_VERSION 1

/// DELEGATE_VERSION: "X.Y.Z"
/// where:
///   X = Major version number
///   Y = Minor version number
///   Z = Patch version number
#define DELEGATE_VERSION STRINGIFY_VALUE(DELEGATE_MAJOR_VERSION) "." \
                         STRINGIFY_VALUE(DELEGATE_MINOR_VERSION) "." \
                         STRINGIFY_VALUE(DELEGATE_PATCH_VERSION)

} //namespace armnnDelegate