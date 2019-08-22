//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "PacketVersionResolver.hpp"

namespace armnn
{

namespace profiling
{

Version PacketVersionResolver::ResolvePacketVersion(uint32_t packetId) const
{
    // NOTE: For now every packet specification is at version 1.0.0
    return Version(1, 0, 0);
}

} // namespace profiling

} // namespace armnn
