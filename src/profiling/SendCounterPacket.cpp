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

#include <unistd.h>

namespace armnn
{

namespace profiling
{

using boost::numeric_cast;

void SendCounterPacket::SendStreamMetaDataPacket()
{
    throw armnn::UnimplementedException();
}

void SendCounterPacket::SendCounterDirectoryPacket(const Category& category, const std::vector<Counter>& counters)
{
    throw armnn::UnimplementedException();
}

void SendCounterPacket::SendPeriodicCounterCapturePacket(uint64_t timestamp, const std::vector<uint32_t>& counterValues,
                                                         const std::vector<uint16_t>& counterUids)
{
    throw armnn::UnimplementedException();
}

void SendCounterPacket::SendPeriodicCounterSelectionPacket(uint32_t capturePeriod,
                                                           const std::vector<uint16_t>& selectedCounterIds)
{
    uint32_t packetFamily = 0;
    uint32_t packetId = 4;
    uint32_t headerSize = numeric_cast<uint32_t>(2 * sizeof(uint32_t));
    uint32_t bodySize = numeric_cast<uint32_t>((1 * sizeof(uint32_t)) + (selectedCounterIds.size() * sizeof(uint16_t)));
    uint32_t totalSize = headerSize + bodySize;
    uint32_t offset = 0;
    uint32_t reserved = 0;

    unsigned char* writeBuffer = m_Buffer.Reserve(totalSize, reserved);

    if (reserved < totalSize)
    {
        // Cancel the operation.
        m_Buffer.Commit(0);
        throw RuntimeException(boost::str(boost::format("No space left in buffer. Unable to reserve (%1%) bytes.")
                               % totalSize));
    }

    if (writeBuffer == nullptr)
    {
        // Cancel the operation.
        m_Buffer.Commit(0);
        throw RuntimeException("Error reserving buffer memory.");
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