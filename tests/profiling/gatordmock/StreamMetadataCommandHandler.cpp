//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "StreamMetadataCommandHandler.hpp"

#include <common/include/CommonProfilingUtils.hpp>

#include <iostream>
#include <sstream>

namespace armnn
{

namespace gatordmock
{

void StreamMetadataCommandHandler::operator()(const arm::pipe::Packet& packet)
{
    ParseData(packet);

    if (m_QuietOperation)
    {
        return;
    }

    std::stringstream ss;

    ss << "Stream metadata packet received" << std::endl << std::endl;

    ss << "Pipe magic: "              << m_PipeMagic                 << std::endl;
    ss << "Stream metadata version: " << m_StreamMetadataVersion     << std::endl;
    ss << "Max data len: "            << m_MaxDataLen                << std::endl;
    ss << "Pid: "                     << m_Pid                       << std::endl;
    ss << "Software info: "           << m_SoftwareInfo              << std::endl;
    ss << "Hardware version: "        << m_HardwareVersion           << std::endl;
    ss << "Software version: "        << m_SoftwareVersion           << std::endl;
    ss << "Process name: "            << m_ProcessName               << std::endl;
    ss << "Packet versions: "         << m_PacketVersionTable.size() << std::endl;

    for (const auto& packetVersion : m_PacketVersionTable)
    {
        ss << "-----------------------" << std::endl;
        ss << "Packet family: "  << packetVersion.m_PacketFamily  << std::endl;
        ss << "Packet id: "      << packetVersion.m_PacketId      << std::endl;
        ss << "Packet version: " << packetVersion.m_PacketVersion << std::endl;
    }

    std::cout << ss.str() << std::endl;
}

std::string ReadString(const unsigned char* buffer, unsigned int &offset)
{
    const char* stringPtr = reinterpret_cast<const char*>(&buffer[offset]);
    return stringPtr != nullptr ? std::string(stringPtr) : "";
}

void StreamMetadataCommandHandler::ParseData(const arm::pipe::Packet &packet)
{
    // Check that at least the packet contains the fixed-length fields
    if (packet.GetLength() < 80)
    {
        return;
    }

    // Utils
    unsigned int uint16_t_size = sizeof(uint16_t);
    unsigned int uint32_t_size = sizeof(uint32_t);

    const unsigned char* buffer = packet.GetData();
    unsigned int offset = 0;

    // Get the fixed-length fields
    m_PipeMagic = arm::pipe::ReadUint32(buffer, offset);
    offset += uint32_t_size;
    m_StreamMetadataVersion = arm::pipe::ReadUint32(buffer, offset);
    offset += uint32_t_size;
    m_MaxDataLen = arm::pipe::ReadUint32(buffer, offset);
    offset += uint32_t_size;
    m_Pid = arm::pipe::ReadUint32(buffer, offset);
    offset += uint32_t_size;
    m_OffsetInfo = arm::pipe::ReadUint32(buffer, offset);
    offset += uint32_t_size;
    m_OffsetHwVersion = arm::pipe::ReadUint32(buffer, offset);
    offset += uint32_t_size;
    m_OffsetSwVersion = arm::pipe::ReadUint32(buffer, offset);
    offset += uint32_t_size;
    m_OffsetProcessName = arm::pipe::ReadUint32(buffer, offset);
    offset += uint32_t_size;
    m_OffsetPacketVersionTable = arm::pipe::ReadUint32(buffer, offset);
    offset += uint32_t_size * 2; // Also skipping the reserved word (all zeros)

    // Get the string fields
    m_SoftwareInfo    = m_OffsetInfo        > 0 ? ReadString(buffer, m_OffsetInfo)        : "";
    m_HardwareVersion = m_OffsetHwVersion   > 0 ? ReadString(buffer, m_OffsetHwVersion)   : "";
    m_SoftwareVersion = m_OffsetSwVersion   > 0 ? ReadString(buffer, m_OffsetSwVersion)   : "";
    m_ProcessName     = m_OffsetProcessName > 0 ? ReadString(buffer, m_OffsetProcessName) : "";

    // Get the packet versions
    m_PacketVersionTable.clear();
    if (m_OffsetPacketVersionTable > 0)
    {
        offset = m_OffsetPacketVersionTable;
        uint16_t packetEntries = arm::pipe::ReadUint16(buffer, offset + uint16_t_size);
        offset += uint32_t_size; // Also skipping the reserved bytes (all zeros)
        for (uint16_t i = 0; i < packetEntries; i++)
        {
            uint16_t packetFamilyAndId = arm::pipe::ReadUint16(buffer, offset + uint16_t_size);
            uint16_t packetFamily = (packetFamilyAndId >> 10) & 0x003F;
            uint16_t packetId     = (packetFamilyAndId >>  0) & 0x03FF;
            offset += uint32_t_size; // Also skipping the reserved bytes (all zeros)
            uint32_t packetVersion = arm::pipe::ReadUint32(buffer, offset);
            offset += uint32_t_size;

            m_PacketVersionTable.push_back({ packetFamily, packetId, packetVersion });
        }
    }
}

} // namespace gatordmock

} // namespace armnn
