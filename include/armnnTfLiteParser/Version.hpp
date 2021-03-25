//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

namespace armnnTfLiteParser
{

/// Macro utils
#define STRINGIFY_VALUE(s) STRINGIFY_MACRO(s)
#define STRINGIFY_MACRO(s) #s

// TfLiteParser version components
#define TFLITE_PARSER_MAJOR_VERSION 24
#define TFLITE_PARSER_MINOR_VERSION 1
#define TFLITE_PARSER_PATCH_VERSION 0

/// TFLITE_PARSER_VERSION: "X.Y.Z"
/// where:
///   X = Major version number
///   Y = Minor version number
///   Z = Patch version number
#define TFLITE_PARSER_VERSION STRINGIFY_VALUE(TFLITE_PARSER_MAJOR_VERSION) "." \
                              STRINGIFY_VALUE(TFLITE_PARSER_MINOR_VERSION) "." \
                              STRINGIFY_VALUE(TFLITE_PARSER_PATCH_VERSION)

} //namespace armnnTfLiteParser
