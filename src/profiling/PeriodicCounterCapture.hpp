//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "IPeriodicCounterCapture.hpp"
#include "Holder.hpp"
#include "Packet.hpp"
#include "SendCounterPacket.hpp"
#include "ICounterValues.hpp"

#include <atomic>
#include <mutex>
#include <thread>

namespace armnn
{

namespace profiling
{

class PeriodicCounterCapture final : public IPeriodicCounterCapture
{
public:
    PeriodicCounterCapture(const Holder& data, ISendCounterPacket& packet, const IReadCounterValues& readCounterValue)
        : m_CaptureDataHolder(data)
        , m_IsRunning(false)
        , m_KeepRunning(false)
        , m_ReadCounterValues(readCounterValue)
        , m_SendCounterPacket(packet)
    {}
    ~PeriodicCounterCapture() { Stop(); }

    void Start() override;
    void Stop() override;
    bool IsRunning() const { return m_IsRunning; }

private:
    CaptureData ReadCaptureData();
    void Capture(const IReadCounterValues& readCounterValues);

    const Holder&             m_CaptureDataHolder;
    bool                      m_IsRunning;
    std::atomic<bool>         m_KeepRunning;
    std::thread               m_PeriodCaptureThread;
    const IReadCounterValues& m_ReadCounterValues;
    ISendCounterPacket&       m_SendCounterPacket;
};

} // namespace profiling

} // namespace armnn
