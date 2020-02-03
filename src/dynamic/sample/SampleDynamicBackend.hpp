//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <cstdint>

extern "C"
{
const char* GetBackendId();
void GetVersion(uint32_t* outMajor, uint32_t* outMinor);
void* BackendFactory();
}
