//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "PeriodicCounterSelectionCommandHandler.hpp"
#include "ProfilingUtils.hpp"

#include <boost/numeric/conversion/cast.hpp>

namespace armnn
{

namespace profiling
{

using namespace std;
using boost::numeric_cast;

void PeriodicCounterSelectionCommandHandler::ParseData(const Packet& packet, CaptureData& captureData)
{
    std::vector<uint16_t> counterIds;
    uint32_t sizeOfUint32 = numeric_cast<uint32_t>(sizeof(uint32_t));
    uint32_t sizeOfUint16 = numeric_cast<uint32_t>(sizeof(uint16_t));
    uint32_t offset = 0;

    if (packet.GetLength() > 0)
    {
        if (packet.GetLength() >= 4)
        {
            captureData.SetCapturePeriod(ReadUint32(reinterpret_cast<const unsigned char*>(packet.GetData()), offset));

            unsigned int counters = (packet.GetLength() - 4) / 2;

            if (counters > 0)
            {
                counterIds.reserve(counters);
                offset += sizeOfUint32;
                for(unsigned int pos = 0; pos < counters; ++pos)
                {
                    counterIds.emplace_back(ReadUint16(reinterpret_cast<const unsigned char*>(packet.GetData()),
                                            offset));
                    offset += sizeOfUint16;
                }
            }

            captureData.SetCounterIds(counterIds);
        }
    }
}

void PeriodicCounterSelectionCommandHandler::operator()(const Packet& packet)
{
    CaptureData captureData;

    ParseData(packet, captureData);

    vector<uint16_t> counterIds = captureData.GetCounterIds();

    m_CaptureDataHolder.SetCaptureData(captureData.GetCapturePeriod(), counterIds);

    m_CaptureThread.Start();

    // Write packet to Counter Stream Buffer
    m_SendCounterPacket.SendPeriodicCounterSelectionPacket(captureData.GetCapturePeriod(), captureData.GetCounterIds());
}

} // namespace profiling

} // namespace armnn