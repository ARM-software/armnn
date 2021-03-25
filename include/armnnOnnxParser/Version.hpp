//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

namespace armnnOnnxParser
{

/// Macro utils
#define STRINGIFY_VALUE(s) STRINGIFY_MACRO(s)
#define STRINGIFY_MACRO(s) #s

// OnnxParser version components
#define ONNX_PARSER_MAJOR_VERSION 24
#define ONNX_PARSER_MINOR_VERSION 1
#define ONNX_PARSER_PATCH_VERSION 0

/// ONNX_PARSER_VERSION: "X.Y.Z"
/// where:
///   X = Major version number
///   Y = Minor version number
///   Z = Patch version number
#define ONNX_PARSER_VERSION STRINGIFY_VALUE(ONNX_PARSER_MAJOR_VERSION) "." \
                            STRINGIFY_VALUE(ONNX_PARSER_MINOR_VERSION) "." \
                            STRINGIFY_VALUE(ONNX_PARSER_PATCH_VERSION)

} //namespace armnnOnnxParser