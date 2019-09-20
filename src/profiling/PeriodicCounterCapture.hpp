//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "Holder.hpp"
#include "IPeriodicCounterCapture.hpp"
#include "Packet.hpp"
#include "IReadCounterValue.hpp"
#include "SendCounterPacket.hpp"

#include "WallClockTimer.hpp"

#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>

namespace armnn
{

namespace profiling
{

class PeriodicCounterCapture final : public IPeriodicCounterCapture
{
public:
    PeriodicCounterCapture(const Holder& data, ISendCounterPacket& packet, const IReadCounterValue& readCounterValue);

    void Start() override;
    void Join();

private:
    CaptureData ReadCaptureData();
    void Functionality(const IReadCounterValue& readCounterValue);

    const Holder&            m_CaptureDataHolder;
    std::atomic<bool>        m_IsRunning;
    std::thread              m_PeriodCaptureThread;
    const IReadCounterValue& m_ReadCounterValue;
    ISendCounterPacket&      m_SendCounterPacket;
};

} // namespace profiling

} // namespace armnn
