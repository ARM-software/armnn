//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

namespace armnnTfParser
{

/// Macro utils
#define STRINGIFY_VALUE(s) STRINGIFY_MACRO(s)
#define STRINGIFY_MACRO(s) #s

// tfParser version components
#define TF_PARSER_MAJOR_VERSION 23
#define TF_PARSER_MINOR_VERSION 0
#define TF_PARSER_PATCH_VERSION 0

/// TF_PARSER_VERSION: "X.Y.Z"
/// where:
///   X = Major version number
///   Y = Minor version number
///   Z = Patch version number
#define TF_PARSER_VERSION STRINGIFY_VALUE(TF_PARSER_MAJOR_VERSION) "." \
                          STRINGIFY_VALUE(TF_PARSER_MINOR_VERSION) "." \
                          STRINGIFY_VALUE(TF_PARSER_PATCH_VERSION)

} //namespace armnnTfParser
