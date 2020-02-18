//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

// Macro utils
#define STRINGIFY_VALUE(s) STRINGIFY_MACRO(s)
#define STRINGIFY_MACRO(s) #s
#define CONCAT_VALUE(a, b, c) CONCAT_MACRO(a, b, c)
#define CONCAT_MACRO(a, b, c) a ## b ## c

// ArmNN version components
#define ARMNN_MAJOR_VERSION 20
#define ARMNN_MINOR_VERSION 02
#define ARMNN_PATCH_VERSION 00

// ARMNN_VERSION: "YYYYMMPP"
// where:
//   YYYY = 4-digit year number
//   MM   = 2-digit month number
//   PP   = 2-digit patch number
#define ARMNN_VERSION "20" STRINGIFY_VALUE(CONCAT_VALUE(ARMNN_MAJOR_VERSION, ARMNN_MINOR_VERSION, ARMNN_PATCH_VERSION))
