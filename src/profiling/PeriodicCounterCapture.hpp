//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "IPeriodicCounterCapture.hpp"
#include "Holder.hpp"
#include "SendCounterPacket.hpp"
#include "ICounterValues.hpp"
#include "CounterIdMap.hpp"

#include <armnn/backends/profiling/IBackendProfilingContext.hpp>

#include <common/include/Packet.hpp>

#include <atomic>
#include <mutex>
#include <thread>

namespace arm
{

namespace pipe
{

class PeriodicCounterCapture final : public IPeriodicCounterCapture
{
public:
    PeriodicCounterCapture(const Holder& data,
                           ISendCounterPacket& packet,
                           IReadCounterValues& readCounterValue,
                           const ICounterMappings& counterIdMap,
                           const std::unordered_map<std::string,
                               std::shared_ptr<IBackendProfilingContext>>& backendProfilingContexts)
            : m_CaptureDataHolder(data)
            , m_IsRunning(false)
            , m_KeepRunning(false)
            , m_ReadCounterValues(readCounterValue)
            , m_SendCounterPacket(packet)
            , m_CounterIdMap(counterIdMap)
            , m_BackendProfilingContexts(backendProfilingContexts)
    {}
    ~PeriodicCounterCapture() { Stop(); }

    void Start() override;
    void Stop() override;
    bool IsRunning() const { return m_IsRunning; }

private:
    CaptureData ReadCaptureData();
    void Capture(IReadCounterValues& readCounterValues);
    void DispatchPeriodicCounterCapturePacket(
            const std::string& backendId, const std::vector<Timestamp>& timestampValues);

    const Holder&             m_CaptureDataHolder;
    bool                      m_IsRunning;
    std::atomic<bool>         m_KeepRunning;
    std::thread               m_PeriodCaptureThread;
    IReadCounterValues&       m_ReadCounterValues;
    ISendCounterPacket&       m_SendCounterPacket;
    const ICounterMappings&   m_CounterIdMap;
    const std::unordered_map<std::string,
            std::shared_ptr<IBackendProfilingContext>>& m_BackendProfilingContexts;
};

} // namespace pipe

} // namespace arm
