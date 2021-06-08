//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "CounterIdMap.hpp"
#include "Holder.hpp"
#include "IProfilingServiceStatus.hpp"
#include "ISendCounterPacket.hpp"

#include <common/include/ProfilingGuidGenerator.hpp>

namespace armnn
{

namespace profiling
{

class IProfilingService : public IProfilingGuidGenerator, public IProfilingServiceStatus
{
public:
    virtual ~IProfilingService() {};
    virtual std::unique_ptr<ISendTimelinePacket> GetSendTimelinePacket() const = 0;
    virtual const ICounterMappings& GetCounterMappings() const = 0;
    virtual ISendCounterPacket& GetSendCounterPacket() = 0;
    virtual bool IsProfilingEnabled() const = 0;
    virtual CaptureData GetCaptureData() = 0;
};

} // namespace profiling

} // namespace armnn

