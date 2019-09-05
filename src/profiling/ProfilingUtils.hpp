//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Exceptions.hpp>

#include <string>
#include <stdint.h>

namespace armnn
{

namespace profiling
{

uint16_t GetNextUid();

void WriteUint64(unsigned char* buffer, unsigned int offset, uint64_t value);

void WriteUint32(unsigned char* buffer, unsigned int offset, uint32_t value);

void WriteUint16(unsigned char* buffer, unsigned int offset, uint16_t value);

uint64_t ReadUint64(const unsigned char* buffer, unsigned int offset);

uint32_t ReadUint32(const unsigned char* buffer, unsigned int offset);

uint16_t ReadUint16(const unsigned char* buffer, unsigned int offset);

std::string GetSoftwareInfo();

std::string GetSoftwareVersion();

std::string GetHardwareVersion();

std::string GetProcessName();

class BufferExhaustion : public armnn::Exception
{
    using Exception::Exception;
};

} // namespace profiling

} // namespace armnn
