//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "MockUtils.hpp"

namespace armnn
{

namespace gatordmock
{

std::string CentreAlignFormatting(const std::string& stringToPass, const int spacingWidth)
{
    std::stringstream outputStream, centrePadding;
    int padding = spacingWidth - static_cast<int>(stringToPass.size());

    for (int i = 0; i < padding / 2; ++i)
    {
        centrePadding << " ";
    }

    outputStream << centrePadding.str() << stringToPass << centrePadding.str();

    if (padding > 0 && padding %2 != 0)
    {
        outputStream << " ";
    }

    return outputStream.str();
}

std::string GetStringNameFromBuffer(const unsigned char* const data, uint32_t offset)
{
    std::string deviceName;
    u_char nextChar = profiling::ReadUint8(data, offset);

    while (IsValidChar(nextChar))
    {
        deviceName += static_cast<char>(nextChar);
        offset ++;
        nextChar = profiling::ReadUint8(data, offset);
    }

    return deviceName;
}

bool IsValidChar(unsigned char c)
{
    // Check that the given character has ASCII 7-bit encoding, alpha-numeric, whitespace, and underscore only
    return c < 128 && (std::isalnum(c) || c == '_' || c == ' ');
}

} // gatordmock

} // armnn
