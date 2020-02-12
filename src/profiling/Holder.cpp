//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/BackendId.hpp>
#include "Holder.hpp"

namespace armnn
{

namespace profiling
{

CaptureData& CaptureData::operator=(const CaptureData& other)
{
    m_CapturePeriod  = other.m_CapturePeriod;
    m_CounterIds     = other.m_CounterIds;
    m_ActiveBackends = other.m_ActiveBackends;

    return *this;
}

void CaptureData::SetActiveBackends(const std::set<armnn::BackendId>& activeBackends)
{
    m_ActiveBackends = activeBackends;
}

void CaptureData::SetCapturePeriod(uint32_t capturePeriod)
{
    m_CapturePeriod = capturePeriod;
}

void CaptureData::SetCounterIds(const std::vector<uint16_t>& counterIds)
{
    m_CounterIds = counterIds;
}

const std::set<armnn::BackendId>& CaptureData::GetActiveBackends() const
{
    return m_ActiveBackends;
}

uint32_t CaptureData::GetCapturePeriod() const
{
    return m_CapturePeriod;
}

const std::vector<uint16_t>& CaptureData::GetCounterIds() const
{
    return m_CounterIds;
}

CaptureData Holder::GetCaptureData() const
{
    std::lock_guard<std::mutex> lockGuard(m_CaptureThreadMutex);

    return m_CaptureData;
}

bool CaptureData::IsCounterIdInCaptureData(uint16_t counterId)
{
    for (auto m_CounterId : m_CounterIds) {
        if (m_CounterId == counterId)
        {
            return true;
        }
    }

    // Return false by default unless counterId is found
    return false;
}

void Holder::SetCaptureData(uint32_t capturePeriod,
                            const std::vector<uint16_t>& counterIds,
                            const std::set<armnn::BackendId>& activeBackends)
{
    std::lock_guard<std::mutex> lockGuard(m_CaptureThreadMutex);

    m_CaptureData.SetCapturePeriod(capturePeriod);
    m_CaptureData.SetCounterIds(counterIds);
    m_CaptureData.SetActiveBackends(activeBackends);

}

} // namespace profiling

} // namespace armnn
