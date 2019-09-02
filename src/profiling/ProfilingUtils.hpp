//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <stdint.h>

namespace armnn
{

namespace profiling
{

void WriteUint32(unsigned char* buffer, unsigned int offset, uint32_t value);

void WriteUint16(unsigned char* buffer, unsigned int offset, uint16_t value);

uint32_t ReadUint32(const unsigned char* buffer, unsigned int offset);

uint16_t ReadUint16(const unsigned char* buffer, unsigned int offset);

} // namespace profiling

} // namespace armnn