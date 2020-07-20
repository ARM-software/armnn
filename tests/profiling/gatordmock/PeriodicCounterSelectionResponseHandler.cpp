//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "PeriodicCounterSelectionResponseHandler.hpp"

#include <common/include/CommonProfilingUtils.hpp>

#include <iostream>

namespace armnn
{

namespace gatordmock
{

void PeriodicCounterSelectionResponseHandler::operator()(const arm::pipe::Packet& packet)
{
    if (!m_QuietOperation)    // Are we supposed to print to stdout?
    {
        uint32_t period = arm::pipe::ReadUint32(reinterpret_cast<const unsigned char*>(packet.GetData()), 0);
        uint32_t numCounters = 0;
        // First check if there are any counters mentioned.
        if(packet.GetLength() > 4)
        {
            // Length will be 4 bytes for the period and then a list of 16 bit UIDS.
            numCounters = ((packet.GetLength() - 4) / 2);
        }
        std::cout << "PeriodicCounterSelectionResponse: Collection interval = " << std::dec << period << "uSec"
                  << " Num counters activated = " << numCounters << std::endl;
    }
}

}    // namespace gatordmock

}    // namespace armnn