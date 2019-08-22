//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "EncodeVersion.hpp"

namespace armnn
{

namespace profiling
{

class PacketVersionResolver final
{
public:
    PacketVersionResolver()  = default;
    ~PacketVersionResolver() = default;

    Version ResolvePacketVersion(uint32_t packetId) const;
};

} // namespace profiling

} // namespace armnn
