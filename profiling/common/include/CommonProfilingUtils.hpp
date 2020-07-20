//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <cstdint>
#include <string>

namespace arm
{

namespace pipe
{
void ReadBytes(const unsigned char* buffer, unsigned int offset, unsigned int valueSize, uint8_t outValue[]);

uint64_t ReadUint64(unsigned const char* buffer, unsigned int offset);

uint32_t ReadUint32(unsigned const char* buffer, unsigned int offset);

uint16_t ReadUint16(unsigned const char* buffer, unsigned int offset);

uint8_t ReadUint8(unsigned const char* buffer, unsigned int offset);

void WriteBytes(unsigned char* buffer, unsigned int offset, const void* value, unsigned int valueSize);

void WriteUint64(unsigned char* buffer, unsigned int offset, uint64_t value);

void WriteUint32(unsigned char* buffer, unsigned int offset, uint32_t value);

void WriteUint16(unsigned char* buffer, unsigned int offset, uint16_t value);

void WriteUint8(unsigned char* buffer, unsigned int offset, uint8_t value);

std::string CentreAlignFormatting(const std::string& stringToPass, const int spacingWidth);

} // namespace pipe
} // namespace arm