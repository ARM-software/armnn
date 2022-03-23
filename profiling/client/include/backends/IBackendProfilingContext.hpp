//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "IBackendProfiling.hpp"

#include <common/include/Optional.hpp>

#include <vector>

namespace arm
{
namespace pipe
{

class IBackendProfilingContext
{
public:
    virtual ~IBackendProfilingContext()
    {}
    virtual uint16_t RegisterCounters(uint16_t currentMaxGlobalCounterID) = 0;
    virtual arm::pipe::Optional<std::string> ActivateCounters(
        uint32_t capturePeriod, const std::vector<uint16_t>& counterIds) = 0;
    virtual std::vector<arm::pipe::Timestamp> ReportCounterValues() = 0;
    virtual bool EnableProfiling(bool flag) = 0;
    virtual bool EnableTimelineReporting(bool flag) = 0;
};

using IBackendProfilingContextUniquePtr = std::unique_ptr<IBackendProfilingContext>;
}    // namespace pipe
}    // namespace arm
