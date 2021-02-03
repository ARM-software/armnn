//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

namespace armnnCaffeParser
{

/// Macro utils
#define STRINGIFY_VALUE(s) STRINGIFY_MACRO(s)
#define STRINGIFY_MACRO(s) #s

// CaffeParser version components
#define CAFFE_PARSER_MAJOR_VERSION 23
#define CAFFE_PARSER_MINOR_VERSION 0
#define CAFFE_PARSER_PATCH_VERSION 0

/// CAFFE_PARSER_VERSION: "X.Y.Z"
/// where:
///   X = Major version number
///   Y = Minor version number
///   Z = Patch version number
#define CAFFE_PARSER_VERSION STRINGIFY_VALUE(CAFFE_PARSER_MAJOR_VERSION) "." \
                             STRINGIFY_VALUE(CAFFE_PARSER_MINOR_VERSION) "." \
                             STRINGIFY_VALUE(CAFFE_PARSER_PATCH_VERSION)

} //namespace armnnCaffeParser