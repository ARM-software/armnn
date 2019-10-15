//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#if !defined(ARMNN_VERSION_FROM_FILE)
#error "A valid version of ArmNN must be provided at compile time"
#endif

#define STRINGIFY_VALUE(s) STRINGIFY_MACRO(s)
#define STRINGIFY_MACRO(s) #s

// YYYYMMPP
// where:
//   YYYY = 4-digit year number
//   MM   = 2-digit month number
//   PP   = 2-digit patch number
// Defined in ArmnnVersion.txt
#define ARMNN_VERSION STRINGIFY_VALUE(ARMNN_VERSION_FROM_FILE)

// Check that the provided ArmNN version is valid
static_assert(sizeof(ARMNN_VERSION) == 9, "Invalid ArmNN version, a valid version should have exactly 8 digits");
