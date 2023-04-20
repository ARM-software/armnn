//
// Copyright © 2017, 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <mutex>
#include <vector>
#include <set>
#include <string>

namespace arm
{

namespace pipe
{

class CaptureData
{
public:
    CaptureData()
        : m_CapturePeriod(0)
        , m_CounterIds()
        , m_ActiveBackends(){}
    CaptureData(uint32_t capturePeriod, std::vector<uint16_t>& counterIds, std::set<std::string> activeBackends)
        : m_CapturePeriod(capturePeriod)
        , m_CounterIds(counterIds)
        , m_ActiveBackends(activeBackends){}
    CaptureData(const CaptureData& captureData)
        : m_CapturePeriod(captureData.m_CapturePeriod)
        , m_CounterIds(captureData.m_CounterIds)
        , m_ActiveBackends(captureData.m_ActiveBackends){}

    CaptureData& operator=(const CaptureData& other);

    void SetActiveBackends(const std::set<std::string>& activeBackends);
    void SetCapturePeriod(uint32_t capturePeriod);
    void SetCounterIds(const std::vector<uint16_t>& counterIds);
    uint32_t GetCapturePeriod() const;
    const std::vector<uint16_t>& GetCounterIds() const;
    const std::set<std::string>& GetActiveBackends() const;
    bool IsCounterIdInCaptureData(uint16_t counterId);

private:
    uint32_t m_CapturePeriod;
    std::vector<uint16_t> m_CounterIds;
    std::set<std::string> m_ActiveBackends;
};

class Holder
{
public:
    Holder()
        : m_CaptureData() {}
    CaptureData GetCaptureData() const;
    void SetCaptureData(uint32_t capturePeriod,
                        const std::vector<uint16_t>& counterIds,
                        const std::set<std::string>& activeBackends);

private:
#if !defined(ARMNN_DISABLE_THREADS)
    mutable std::mutex m_CaptureThreadMutex;
#endif
    CaptureData m_CaptureData;
};

} // namespace pipe

} // namespace arm
