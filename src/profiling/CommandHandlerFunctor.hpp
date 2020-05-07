//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "common/include/Packet.hpp"

#include <armnn/utility/IgnoreUnused.hpp>

#include <cstdint>

namespace armnn
{

namespace profiling
{

class CommandHandlerFunctor
{
public:
    CommandHandlerFunctor(uint32_t familyId, uint32_t packetId, uint32_t version)
        : m_FamilyId(familyId),
          m_PacketId(packetId)
        , m_Version(version)
    {}

    uint32_t GetFamilyId() const;
    uint32_t GetPacketId() const;
    uint32_t GetVersion()  const;

    virtual void operator()(const Packet& packet) = 0;

    virtual ~CommandHandlerFunctor() {}

private:
    uint32_t m_FamilyId;
    uint32_t m_PacketId;
    uint32_t m_Version;
};

} // namespace profiling

} // namespace armnn
