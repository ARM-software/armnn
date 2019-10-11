//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <EncodeVersion.hpp>

namespace armnn
{

namespace gatordmock
{


uint32_t ConstructHeader(uint32_t packetFamily,
                         uint32_t packetClass,
                         uint32_t packetType)
{
    return ((packetFamily & 0x3F) << 26)|
           ((packetClass & 0x3FF) << 19)|
           ((packetType & 0x3FFF) << 16);
}

uint32_t ConstructHeader(uint32_t packetFamily,
                         uint32_t packetId)
{
    return ((packetFamily & 0x3F) << 26)|
           ((packetId & 0x3FF) << 16);
}

} // gatordmock

} // armnn
