//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <mutex>
#include <vector>

namespace armnn
{

namespace profiling
{

class CaptureData
{
public:
    CaptureData()
        : m_CapturePeriod(0)
        , m_CounterIds() {}
    CaptureData(uint32_t capturePeriod, std::vector<uint16_t>& counterIds)
        : m_CapturePeriod(capturePeriod)
        , m_CounterIds(counterIds) {}
    CaptureData(const CaptureData& captureData)
        : m_CapturePeriod(captureData.m_CapturePeriod)
        , m_CounterIds(captureData.m_CounterIds) {}

    CaptureData& operator= (const CaptureData& captureData);

    void SetCapturePeriod(uint32_t capturePeriod);
    void SetCounterIds(const std::vector<uint16_t>& counterIds);
    uint32_t GetCapturePeriod() const;
    std::vector<uint16_t> GetCounterIds() const;

private:
    uint32_t m_CapturePeriod;
    std::vector<uint16_t> m_CounterIds;
};

class Holder
{
public:
    Holder()
        : m_CaptureData() {}
    CaptureData GetCaptureData() const;
    void SetCaptureData(uint32_t capturePeriod, const std::vector<uint16_t>& counterIds);

private:
    mutable std::mutex m_CaptureThreadMutex;
    CaptureData m_CaptureData;
};

} // namespace profiling

} // namespace armnn
