//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/IRuntime.hpp>
#include <vector>

namespace armnn
{
namespace profiling
{

class IBackendProfilingContext
{
protected:
    IBackendProfilingContext(const IRuntime::CreationOptions&)
    {}

public:
    virtual ~IBackendProfilingContext()
    {}
    virtual uint16_t RegisterCounters(uint16_t currentMaxGlobalCounterID);
    virtual void ActivateCounters(uint32_t capturePeriod, const std::vector<uint16_t>& counterIds);
    virtual std::vector<Timestamp> ReportCounterValues();
    virtual void EnableProfiling(bool flag);
};

using IBackendProfilingContextUniquePtr = std::unique_ptr<IBackendProfilingContext>;
}    // namespace profiling
}    // namespace armnn