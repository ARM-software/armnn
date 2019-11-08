//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <Packet.hpp>
#include <CommandHandlerFunctor.hpp>

#include <vector>

namespace armnn
{

namespace gatordmock
{

struct PacketVersion
{
    uint16_t m_PacketFamily;
    uint16_t m_PacketId;
    uint32_t m_PacketVersion;
};

class StreamMetadataCommandHandler : public profiling::CommandHandlerFunctor
{

public:
    /**
     * @param familyId The family of the packets this handler will service
     * @param packetId The id of packets this handler will process
     * @param version The version of that id
     * @param quietOperation Optional parameter to turn off printouts. This is useful for unit tests
     */
    StreamMetadataCommandHandler(uint32_t familyId,
                                 uint32_t packetId,
                                 uint32_t version,
                                 bool quietOperation = false)
        : CommandHandlerFunctor(familyId, packetId, version)
        , m_QuietOperation(quietOperation)
    {}

    void operator()(const armnn::profiling::Packet& packet) override;

private:
    void ParseData(const armnn::profiling::Packet& packet);

    uint32_t m_PipeMagic;
    uint32_t m_StreamMetadataVersion;
    uint32_t m_MaxDataLen;
    uint32_t m_Pid;
    uint32_t m_OffsetInfo;
    uint32_t m_OffsetHwVersion;
    uint32_t m_OffsetSwVersion;
    uint32_t m_OffsetProcessName;
    uint32_t m_OffsetPacketVersionTable;

    std::string m_SoftwareInfo;
    std::string m_HardwareVersion;
    std::string m_SoftwareVersion;
    std::string m_ProcessName;

    std::vector<PacketVersion> m_PacketVersionTable;

    bool m_QuietOperation;
};

} // namespace gatordmock

} // namespace armnn
