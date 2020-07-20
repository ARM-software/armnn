//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <common/include/CommonProfilingUtils.hpp>
#include <common/include/NumericCast.hpp>
#include <common/include/ProfilingException.hpp>
#include <common/include/SwTrace.hpp>

#include <sstream>

namespace arm
{

namespace pipe
{

// Calculate the actual length an SwString will be including the terminating null character
// padding to bring it to the next uint32_t boundary but minus the leading uint32_t encoding
// the size to allow the offset to be correctly updated when decoding a binary packet.
uint32_t CalculateSizeOfPaddedSwString(const std::string& str)
{
    std::vector<uint32_t> swTraceString;
    StringToSwTraceString<SwTraceCharPolicy>(str, swTraceString);
    unsigned int uint32_t_size = sizeof(uint32_t);
    uint32_t size = (numeric_cast<uint32_t>(swTraceString.size()) - 1) * uint32_t_size;
    return size;
}

// Read TimelineMessageDirectoryPacket from given IPacketBuffer and offset
SwTraceMessage ReadSwTraceMessage(const unsigned char* packetBuffer,
                                  unsigned int& offset,
                                  const unsigned int& packetLength)
{
    ARM_PIPE_ASSERT(packetBuffer);

    unsigned int uint32_t_size = sizeof(uint32_t);

    SwTraceMessage swTraceMessage;

    // Read the decl_id
    uint32_t readDeclId = ReadUint32(packetBuffer, offset);
    swTraceMessage.m_Id = readDeclId;

    // SWTrace "namestring" format
    // length of the string (first 4 bytes) + string + null terminator

    // Check the decl_name
    offset += uint32_t_size;
    uint32_t swTraceDeclNameLength = ReadUint32(packetBuffer, offset);

    if (swTraceDeclNameLength == 0 || swTraceDeclNameLength > packetLength)
    {
        throw arm::pipe::ProfilingException("Error swTraceDeclNameLength is an invalid size", LOCATION());
    }

    offset += uint32_t_size;
    std::vector<unsigned char> swTraceStringBuffer(swTraceDeclNameLength - 1);
    std::memcpy(swTraceStringBuffer.data(),
                packetBuffer + offset, swTraceStringBuffer.size());

    swTraceMessage.m_Name.assign(swTraceStringBuffer.begin(), swTraceStringBuffer.end()); // name

    // Check the ui_name
    offset += CalculateSizeOfPaddedSwString(swTraceMessage.m_Name);
    uint32_t swTraceUINameLength = ReadUint32(packetBuffer, offset);

    if (swTraceUINameLength == 0 || swTraceUINameLength > packetLength)
    {
        throw arm::pipe::ProfilingException("Error swTraceUINameLength is an invalid size", LOCATION());
    }

    offset += uint32_t_size;
    swTraceStringBuffer.resize(swTraceUINameLength - 1);
    std::memcpy(swTraceStringBuffer.data(),
                packetBuffer  + offset, swTraceStringBuffer.size());

    swTraceMessage.m_UiName.assign(swTraceStringBuffer.begin(), swTraceStringBuffer.end()); // ui_name

    // Check arg_types
    offset += CalculateSizeOfPaddedSwString(swTraceMessage.m_UiName);
    uint32_t swTraceArgTypesLength = ReadUint32(packetBuffer, offset);

    if (swTraceArgTypesLength == 0 || swTraceArgTypesLength > packetLength)
    {
        throw arm::pipe::ProfilingException("Error swTraceArgTypesLength is an invalid size", LOCATION());
    }

    offset += uint32_t_size;
    swTraceStringBuffer.resize(swTraceArgTypesLength - 1);
    std::memcpy(swTraceStringBuffer.data(),
                packetBuffer  + offset, swTraceStringBuffer.size());

    swTraceMessage.m_ArgTypes.assign(swTraceStringBuffer.begin(), swTraceStringBuffer.end()); // arg_types

    std::string swTraceString(swTraceStringBuffer.begin(), swTraceStringBuffer.end());

    // Check arg_names
    offset += CalculateSizeOfPaddedSwString(swTraceString);
    uint32_t swTraceArgNamesLength = ReadUint32(packetBuffer, offset);

    if (swTraceArgNamesLength == 0 || swTraceArgNamesLength > packetLength)
    {
        throw arm::pipe::ProfilingException("Error swTraceArgNamesLength is an invalid size", LOCATION());
    }

    offset += uint32_t_size;
    swTraceStringBuffer.resize(swTraceArgNamesLength - 1);
    std::memcpy(swTraceStringBuffer.data(),
                packetBuffer  + offset, swTraceStringBuffer.size());

    swTraceString.assign(swTraceStringBuffer.begin(), swTraceStringBuffer.end());
    std::stringstream stringStream(swTraceString);
    std::string argName;
    while (std::getline(stringStream, argName, ','))
    {
        swTraceMessage.m_ArgNames.push_back(argName);
    }

    offset += CalculateSizeOfPaddedSwString(swTraceString);

    return swTraceMessage;
}

} // namespace pipe

} // namespace arm
