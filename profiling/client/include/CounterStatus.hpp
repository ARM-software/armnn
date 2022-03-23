//
// Copyright © 2022 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <cstdint>

namespace arm
{

namespace pipe
{

struct CounterStatus
{
    CounterStatus(uint16_t backendCounterId,
                  uint16_t globalCounterId,
                  bool enabled,
                  uint32_t samplingRateInMicroseconds)
                  : m_BackendCounterId(backendCounterId),
                    m_GlobalCounterId(globalCounterId),
                    m_Enabled(enabled),
                    m_SamplingRateInMicroseconds(samplingRateInMicroseconds) {}
    uint16_t m_BackendCounterId;
    uint16_t m_GlobalCounterId;
    bool     m_Enabled;
    uint32_t m_SamplingRateInMicroseconds;
};

}    // namespace pipe

}    // namespace arm
