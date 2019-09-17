//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Holder.hpp"

namespace armnn
{

namespace profiling
{

CaptureData& CaptureData::operator= (const CaptureData& captureData)
{
    m_CapturePeriod = captureData.m_CapturePeriod;
    m_CounterIds    = captureData.m_CounterIds;

    return *this;
}

void CaptureData::SetCapturePeriod(uint32_t capturePeriod)
{
    m_CapturePeriod = capturePeriod;
}

void CaptureData::SetCounterIds(const std::vector<uint16_t>& counterIds)
{
    m_CounterIds = counterIds;
}

std::uint32_t CaptureData::GetCapturePeriod() const
{
    return m_CapturePeriod;
}

std::vector<uint16_t> CaptureData::GetCounterIds() const
{
    return m_CounterIds;
}

CaptureData Holder::GetCaptureData() const
{
    std::lock_guard<std::mutex> lockGuard(m_CaptureThreadMutex);
    return m_CaptureData;
}

void Holder::SetCaptureData(uint32_t capturePeriod, const std::vector<uint16_t>& counterIds)
{
    std::lock_guard<std::mutex> lockGuard(m_CaptureThreadMutex);
    m_CaptureData.SetCapturePeriod(capturePeriod);
    m_CaptureData.SetCounterIds(counterIds);
}

} // namespace profiling

} // namespace armnn
