//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RequestCountersPacketHandler.hpp"

#include "DirectoryCaptureCommandHandler.hpp"

#include <armnn/utility/NumericCast.hpp>

#include <common/include/PacketVersionResolver.hpp>
#include <common/include/ProfilingException.hpp>

namespace armnn
{

namespace profiling
{

std::vector<uint32_t> RequestCountersPacketHandler::GetHeadersAccepted()
{
    std::vector<uint32_t> headers;
    headers.push_back(m_CounterDirectoryMessageHeader); // counter directory
    return headers;
}

void RequestCountersPacketHandler::HandlePacket(const arm::pipe::Packet& packet)
{
    if (packet.GetHeader() != m_CounterDirectoryMessageHeader)
    {
        return;
    }
    arm::pipe::PacketVersionResolver packetVersionResolver;
    DirectoryCaptureCommandHandler directoryCaptureCommandHandler(
            0, 2, packetVersionResolver.ResolvePacketVersion(0, 2).GetEncodedValue());
    directoryCaptureCommandHandler.operator()(packet);
    const ICounterDirectory& counterDirectory = directoryCaptureCommandHandler.GetCounterDirectory();
    for (auto& category : counterDirectory.GetCategories())
    {
        // Remember we need to translate the Uid's from our CounterDirectory instance to the parent one.
        std::vector<uint16_t> translatedCounters;
        for (auto const& copyUid : category->m_Counters)
        {
            translatedCounters.emplace_back(directoryCaptureCommandHandler.TranslateUIDCopyToOriginal(copyUid));
        }
        m_IdList.insert(std::end(m_IdList), std::begin(translatedCounters), std::end(translatedCounters));
    }
    SendCounterSelectionPacket();
}

void RequestCountersPacketHandler::SendCounterSelectionPacket()
{
    uint32_t uint16_t_size = sizeof(uint16_t);
    uint32_t uint32_t_size = sizeof(uint32_t);

    uint32_t offset   = 0;
    uint32_t bodySize = uint32_t_size + armnn::numeric_cast<uint32_t>(m_IdList.size()) * uint16_t_size;

    auto uniqueData     = std::make_unique<unsigned char[]>(bodySize);
    auto data = reinterpret_cast<unsigned char*>(uniqueData.get());

    // Copy capturePeriod
    WriteUint32(data, offset, m_CapturePeriod);

    // Copy m_IdList
    offset += uint32_t_size;
    for (const uint16_t& id : m_IdList)
    {
        WriteUint16(data, offset, id);
        offset += uint16_t_size;
    }

    arm::pipe::Packet packet(0x40000, bodySize, uniqueData);
    m_Connection->ReturnPacket(packet);
}

} // namespace profiling

} // namespace armnn