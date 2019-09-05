//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SendCounterPacket.hpp"
#include "EncodeVersion.hpp"
#include "ProfilingUtils.hpp"

#include <armnn/Exceptions.hpp>

#include <boost/format.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/core/ignore_unused.hpp>

#include <cstring>

namespace armnn
{

namespace profiling
{

using boost::numeric_cast;

const unsigned int SendCounterPacket::PIPE_MAGIC;
const unsigned int SendCounterPacket::MAX_METADATA_PACKET_LENGTH;

void SendCounterPacket::SendStreamMetaDataPacket()
{
    std::string info(GetSoftwareInfo());
    std::string hardwareVersion(GetHardwareVersion());
    std::string softwareVersion(GetSoftwareVersion());
    std::string processName = GetProcessName().substr(0, 60);

    uint32_t infoSize = numeric_cast<uint32_t>(info.size()) > 0 ? numeric_cast<uint32_t>(info.size()) + 1 : 0;
    uint32_t hardwareVersionSize = numeric_cast<uint32_t>(hardwareVersion.size()) > 0 ?
                                   numeric_cast<uint32_t>(hardwareVersion.size()) + 1 : 0;
    uint32_t softwareVersionSize = numeric_cast<uint32_t>(softwareVersion.size()) > 0 ?
                                   numeric_cast<uint32_t>(softwareVersion.size()) + 1 : 0;
    uint32_t processNameSize = numeric_cast<uint32_t>(processName.size()) > 0 ?
                               numeric_cast<uint32_t>(processName.size()) + 1 : 0;

    uint32_t sizeUint32 = numeric_cast<uint32_t>(sizeof(uint32_t));

    uint32_t headerSize = 2 * sizeUint32;
    uint32_t bodySize = 10 * sizeUint32;
    uint32_t packetVersionCountSize = sizeUint32;

    // Supported Packets
    // Stream metadata packet            (packet family=0; packet id=0)
    // Connection Acknowledged packet    (packet family=0, packet id=1)
    // Counter Directory packet          (packet family=0; packet id=2)
    // Request Counter Directory packet  (packet family=0, packet id=3)
    // Periodic Counter Selection packet (packet family=0, packet id=4)
    uint32_t packetVersionEntries = 5;

    uint32_t payloadSize = numeric_cast<uint32_t>(infoSize + hardwareVersionSize + softwareVersionSize +
                                                  processNameSize + packetVersionCountSize +
                                                  (packetVersionEntries * 2 * sizeUint32));

    uint32_t totalSize = headerSize + bodySize + payloadSize;
    uint32_t offset = 0;
    uint32_t reserved = 0;

    unsigned char *writeBuffer = m_Buffer.Reserve(totalSize, reserved);

    if (reserved < totalSize)
    {
        CancelOperationAndThrow<BufferExhaustion>(
                    boost::str(boost::format("No space left in buffer. Unable to reserve (%1%) bytes.")
                               % totalSize));
    }

    if (writeBuffer == nullptr)
    {
        CancelOperationAndThrow<RuntimeException>("Error reserving buffer memory.");
    }

    try
    {
        // Create header

        WriteUint32(writeBuffer, offset, 0);
        offset += sizeUint32;
        WriteUint32(writeBuffer, offset, totalSize - headerSize);

        // Packet body

        offset += sizeUint32;
        WriteUint32(writeBuffer, offset, PIPE_MAGIC); // pipe_magic
        offset += sizeUint32;
        WriteUint32(writeBuffer, offset, EncodeVersion(1, 0, 0)); // stream_metadata_version
        offset += sizeUint32;
        WriteUint32(writeBuffer, offset, MAX_METADATA_PACKET_LENGTH); // max_data_length
        offset += sizeUint32;
        WriteUint32(writeBuffer, offset, numeric_cast<uint32_t>(getpid())); // pid
        offset += sizeUint32;
        uint32_t poolOffset = bodySize;
        WriteUint32(writeBuffer, offset, infoSize ? poolOffset : 0); // offset_info
        offset += sizeUint32;
        poolOffset += infoSize;
        WriteUint32(writeBuffer, offset, hardwareVersionSize ? poolOffset : 0); // offset_hw_version
        offset += sizeUint32;
        poolOffset += hardwareVersionSize;
        WriteUint32(writeBuffer, offset, softwareVersionSize ? poolOffset : 0); // offset_sw_version
        offset += sizeUint32;
        poolOffset += softwareVersionSize;
        WriteUint32(writeBuffer, offset, processNameSize ? poolOffset : 0); // offset_process_name
        offset += sizeUint32;
        poolOffset += processNameSize;
        WriteUint32(writeBuffer, offset, packetVersionEntries ? poolOffset : 0); // offset_packet_version_table
        offset += sizeUint32;
        WriteUint32(writeBuffer, offset, 0); // reserved
        offset += sizeUint32;

        // Pool

        if (infoSize)
        {
            memcpy(&writeBuffer[offset], info.c_str(), infoSize);
            offset += infoSize;
        }

        if (hardwareVersionSize)
        {
            memcpy(&writeBuffer[offset], hardwareVersion.c_str(), hardwareVersionSize);
            offset += hardwareVersionSize;
        }

        if (softwareVersionSize)
        {
            memcpy(&writeBuffer[offset], softwareVersion.c_str(), softwareVersionSize);
            offset += softwareVersionSize;
        }

        if (processNameSize)
        {
            memcpy(&writeBuffer[offset], processName.c_str(), processNameSize);
            offset += processNameSize;
        }

        if (packetVersionEntries)
        {
            // Packet Version Count
            WriteUint32(writeBuffer, offset, packetVersionEntries << 16);

            // Packet Version Entries
            uint32_t packetFamily = 0;
            uint32_t packetId = 0;

            offset += sizeUint32;
            for (uint32_t i = 0; i < packetVersionEntries; ++i) {
                WriteUint32(writeBuffer, offset, ((packetFamily & 0x3F) << 26) | ((packetId++ & 0x3FF) << 16));
                offset += sizeUint32;
                WriteUint32(writeBuffer, offset, EncodeVersion(1, 0, 0));
                offset += sizeUint32;
            }
        }
    }
    catch(...)
    {
        CancelOperationAndThrow<RuntimeException>("Error processing packet.");
    }

    m_Buffer.Commit(totalSize);
}

void SendCounterPacket::SendCounterDirectoryPacket(const CounterDirectory& counterDirectory)
{
    throw armnn::UnimplementedException();
}

void SendCounterPacket::SendPeriodicCounterCapturePacket(uint64_t timestamp, const IndexValuePairsVector& values)
{
    uint32_t packetFamily = 1;
    uint32_t packetClass = 0;
    uint32_t packetType = 0;
    uint32_t headerSize = numeric_cast<uint32_t>(2 * sizeof(uint32_t));
    uint32_t bodySize = numeric_cast<uint32_t>((1 * sizeof(uint64_t)) +
                                               (values.size() * (sizeof(uint16_t) + sizeof(uint32_t))));
    uint32_t totalSize = headerSize + bodySize;
    uint32_t offset = 0;
    uint32_t reserved = 0;

    unsigned char* writeBuffer = m_Buffer.Reserve(totalSize, reserved);

    if (reserved < totalSize)
    {
        CancelOperationAndThrow<BufferExhaustion>(
                    boost::str(boost::format("No space left in buffer. Unable to reserve (%1%) bytes.")
                               % totalSize));
    }

    if (writeBuffer == nullptr)
    {
        CancelOperationAndThrow<RuntimeException>("Error reserving buffer memory.");
    }

    // Create header.
    WriteUint32(writeBuffer,
                offset,
                ((packetFamily & 0x3F) << 26) | ((packetClass & 0x3FF) << 19) | ((packetType & 0x3FFF) << 16));
    offset += numeric_cast<uint32_t>(sizeof(uint32_t));
    WriteUint32(writeBuffer, offset, bodySize);

    // Copy captured Timestamp.
    offset += numeric_cast<uint32_t>(sizeof(uint32_t));
    WriteUint64(writeBuffer, offset, timestamp);

    // Copy selectedCounterIds.
    offset += numeric_cast<uint32_t>(sizeof(uint64_t));
    for (const auto& pair: values)
    {
        WriteUint16(writeBuffer, offset, pair.first);
        offset += numeric_cast<uint32_t>(sizeof(uint16_t));
        WriteUint32(writeBuffer, offset, pair.second);
        offset += numeric_cast<uint32_t>(sizeof(uint32_t));
    }

    m_Buffer.Commit(totalSize);
}

void SendCounterPacket::SendPeriodicCounterSelectionPacket(uint32_t capturePeriod,
                                                           const std::vector<uint16_t>& selectedCounterIds)
{
    uint32_t packetFamily = 0;
    uint32_t packetId = 4;
    uint32_t headerSize = numeric_cast<uint32_t>(2 * sizeof(uint32_t));
    uint32_t bodySize   = numeric_cast<uint32_t>((1 * sizeof(uint32_t)) +
                                                 (selectedCounterIds.size() * sizeof(uint16_t)));
    uint32_t totalSize = headerSize + bodySize;
    uint32_t offset = 0;
    uint32_t reserved = 0;

    unsigned char* writeBuffer = m_Buffer.Reserve(totalSize, reserved);

    if (reserved < totalSize)
    {
        CancelOperationAndThrow<BufferExhaustion>(
                    boost::str(boost::format("No space left in buffer. Unable to reserve (%1%) bytes.")
                               % totalSize));
    }

    if (writeBuffer == nullptr)
    {
        CancelOperationAndThrow<RuntimeException>("Error reserving buffer memory.");
    }

    // Create header.
    WriteUint32(writeBuffer, offset, ((packetFamily & 0x3F) << 26) | ((packetId & 0x3FF) << 16));
    offset += numeric_cast<uint32_t>(sizeof(uint32_t));
    WriteUint32(writeBuffer, offset, bodySize);

    // Copy capturePeriod.
    offset += numeric_cast<uint32_t>(sizeof(uint32_t));
    WriteUint32(writeBuffer, offset, capturePeriod);

    // Copy selectedCounterIds.
    offset += numeric_cast<uint32_t>(sizeof(uint32_t));
    for(const uint16_t& id: selectedCounterIds)
    {
        WriteUint16(writeBuffer, offset, id);
        offset += numeric_cast<uint32_t>(sizeof(uint16_t));
    }

    m_Buffer.Commit(totalSize);
}

void SendCounterPacket::SetReadyToRead()
{
    m_ReadyToRead = true;
}

} // namespace profiling

} // namespace armnn
