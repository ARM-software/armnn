//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "PeriodicCounterCaptureCommandHandler.hpp"

#include <ProfilingUtils.hpp>

#include <boost/numeric/conversion/cast.hpp>

#include <iostream>

namespace armnn
{

namespace gatordmock
{

using boost::numeric_cast;

void PeriodicCounterCaptureCommandHandler::ParseData(const armnn::profiling::Packet& packet)
{
    std::vector<uint16_t> counterIds;
    std::vector<uint32_t> counterValues;

    uint32_t sizeOfUint64 = numeric_cast<uint32_t>(sizeof(uint64_t));
    uint32_t sizeOfUint32 = numeric_cast<uint32_t>(sizeof(uint32_t));
    uint32_t sizeOfUint16 = numeric_cast<uint32_t>(sizeof(uint16_t));

    uint32_t offset = 0;

    if (packet.GetLength() >= 8)
    {
        offset = 0;

        uint64_t timestamp = profiling::ReadUint64(reinterpret_cast<const unsigned char*>(packet.GetData()), offset);

        if (m_FirstTimestamp == 0)    // detect the first timestamp we receive.
        {
            m_FirstTimestamp = timestamp;
        }
        else
        {
            m_SecondTimestamp    = timestamp;
            m_CurrentPeriodValue = m_SecondTimestamp - m_FirstTimestamp;
            m_FirstTimestamp     = m_SecondTimestamp;
        }

        // Length minus timestamp and header divided by the length of an indexPair
        unsigned int counters = (packet.GetLength() - 8) / 6;

        if (counters > 0)
        {
            counterIds.reserve(counters);
            counterValues.reserve(counters);
            // Move offset over timestamp area
            offset += sizeOfUint64;
            for (unsigned int pos = 0; pos < counters; ++pos)
            {
                counterIds.emplace_back(
                    profiling::ReadUint16(reinterpret_cast<const unsigned char*>(packet.GetData()), offset));
                offset += sizeOfUint16;

                counterValues.emplace_back(
                    profiling::ReadUint32(reinterpret_cast<const unsigned char*>(packet.GetData()), offset));
                offset += sizeOfUint32;
            }
        }

        m_CounterCaptureValues.m_Timestamp = timestamp;
        m_CounterCaptureValues.m_Uids      = counterIds;
        m_CounterCaptureValues.m_Values    = counterValues;
    }
}

void PeriodicCounterCaptureCommandHandler::operator()(const profiling::Packet& packet)
{
    ParseData(packet);
    if (!m_QuietOperation)    // Are we supposed to print to stdout?
    {
        std::string header, body, uidString, valueString;

        for (uint16_t uid : m_CounterCaptureValues.m_Uids)
        {
            uidString.append(std::to_string(uid));
            uidString.append(", ");
        }

        for (uint32_t val : m_CounterCaptureValues.m_Values)
        {
            valueString.append(std::to_string(val));
            valueString.append(", ");
        }

        body.append(profiling::CentreAlignFormatting(std::to_string(m_CounterCaptureValues.m_Timestamp), 10));
        body.append(" | ");
        body.append(profiling::CentreAlignFormatting(std::to_string(m_CurrentPeriodValue), 13));
        body.append(" | ");
        body.append(profiling::CentreAlignFormatting(uidString, 10));
        body.append(" | ");
        body.append(profiling::CentreAlignFormatting(valueString, 10));
        body.append("\n");

        if (!m_HeaderPrinted)
        {
            header.append(profiling::CentreAlignFormatting(" Timestamp", 11));
            header.append(" | ");
            header.append(profiling::CentreAlignFormatting("Period (us)", 13));
            header.append(" | ");
            header.append(profiling::CentreAlignFormatting("UID's", static_cast<int>(uidString.size())));
            header.append(" | ");
            header.append(profiling::CentreAlignFormatting("Values", 10));
            header.append("\n");

            std::cout << header;
            m_HeaderPrinted = true;
        }

        std::cout << std::string(body.size(), '-') << "\n";

        std::cout << body;
    }
}

}    // namespace gatordmock

}    // namespace armnn
