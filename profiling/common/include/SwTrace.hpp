//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "NumericCast.hpp"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

namespace arm
{

namespace pipe
{

struct SwTraceHeader
{
    uint8_t m_StreamVersion;
    uint8_t m_PointerBytes;
    uint8_t m_ThreadIdBytes;
};

struct SwTraceMessage
{
    uint32_t m_Id;
    std::string m_Name;
    std::string m_UiName;
    std::vector<char> m_ArgTypes;
    std::vector<std::string> m_ArgNames;
};

struct SwTraceCharPolicy
{
    static bool IsValidChar(unsigned char c)
    {
        // Check that the given character has ASCII 7-bit encoding
        return c < 128;
    }
};

struct SwTraceNameCharPolicy
{
    static bool IsValidChar(unsigned char c)
    {
        // Check that the given character has ASCII 7-bit encoding, alpha-numeric and underscore only
        return c < 128 && (std::isalnum(c) || c == '_');
    }
};

struct SwTraceTypeCharPolicy
{
    static bool IsValidChar(unsigned char c)
    {
        // Check that the given character is among the allowed ones
        switch (c)
        {
        case '@':
        case 't':
        case 'i':
        case 'I':
        case 'l':
        case 'L':
        case 'F':
        case 'p':
        case 's':
            return true; // Valid char
        default:
            return false; // Invalid char
        }
    }
};

template <typename SwTracePolicy>
bool IsValidSwTraceString(const std::string& s)
{
    // Check that all the characters in the given string conform to the given policy
    return std::all_of(s.begin(), s.end(), [](unsigned char c) { return SwTracePolicy::IsValidChar(c); });
}

template <typename SwTracePolicy>
bool StringToSwTraceString(const std::string& s, std::vector<uint32_t>& outputBuffer)
{
    // Converts the given string to an SWTrace "string" (i.e. a string of "chars"), and writes it into
    // the given buffer including the null-terminator. It also pads it to the next uint32_t if necessary

    // Clear the output buffer
    outputBuffer.clear();

    // Check that the given string is a valid SWTrace "string" (i.e. a string of "chars")
    if (!IsValidSwTraceString<SwTracePolicy>(s))
    {
        return false;
    }

    // Prepare the output buffer
    size_t s_size        = s.size() + 1;    // The size of the string (in chars) plus the null-terminator
    size_t uint32_t_size = sizeof(uint32_t);
    // Output buffer size = StringLength (32 bit) + amount of complete 32bit words that fit into the string
    //                      + an additional 32bit word if there are remaining chars to complete the string
    //                      (The rest of the 32bit word is then filled with the NULL terminator)
    size_t outBufferSize = 1 + (s_size / uint32_t_size) + (s_size % uint32_t_size != 0 ? 1 : 0);
    outputBuffer.resize(outBufferSize, '\0');

    // Write the SWTrace string to the output buffer
    outputBuffer[0] = numeric_cast<uint32_t>(s_size);
    std::memcpy(outputBuffer.data() + 1, s.data(), s_size);

    return true;
}

template <typename SwTracePolicy,
          typename SwTraceBuffer = std::vector<uint32_t>>
bool ConvertDirectoryComponent(const std::string& directoryComponent, SwTraceBuffer& swTraceBuffer)
{
    // Convert the directory component using the given policy
    SwTraceBuffer tempSwTraceBuffer;
    bool result = StringToSwTraceString<SwTracePolicy>(directoryComponent, tempSwTraceBuffer);
    if (!result)
    {
        return false;
    }

    swTraceBuffer.insert(swTraceBuffer.end(), tempSwTraceBuffer.begin(), tempSwTraceBuffer.end());

    return true;
}

uint32_t CalculateSizeOfPaddedSwString(const std::string& str);

SwTraceMessage ReadSwTraceMessage(const unsigned char*, unsigned int&, const unsigned int& packetLength);

} // namespace pipe

} // namespace arm
