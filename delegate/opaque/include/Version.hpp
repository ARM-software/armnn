//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

namespace armnnOpaqueDelegate
{

/// Macro utils
#define STRINGIFY_VALUE(s) STRINGIFY_MACRO(s)
#define STRINGIFY_MACRO(s) #s

// ArmNN Delegate version components
#define OPAQUE_DELEGATE_MAJOR_VERSION 1
#define OPAQUE_DELEGATE_MINOR_VERSION 0
#define OPAQUE_DELEGATE_PATCH_VERSION 0

/// DELEGATE_VERSION: "X.Y.Z"
/// where:
///   X = Major version number
///   Y = Minor version number
///   Z = Patch version number
#define OPAQUE_DELEGATE_VERSION STRINGIFY_VALUE(OPAQUE_DELEGATE_MAJOR_VERSION) "." \
                                STRINGIFY_VALUE(OPAQUE_DELEGATE_MINOR_VERSION) "." \
                                STRINGIFY_VALUE(OPAQUE_DELEGATE_PATCH_VERSION)

} //namespace armnnDelegate