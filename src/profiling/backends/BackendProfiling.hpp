//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ProfilingService.hpp"
#include <armnn/backends/profiling/IBackendProfiling.hpp>

namespace armnn
{

namespace profiling
{

class BackendProfiling : public IBackendProfiling
{
public:
    BackendProfiling(const IRuntime::CreationOptions& options,
                     ProfilingService& profilingService,
                     const BackendId& backendId)
            : m_Options(options),
              m_ProfilingService(profilingService),
              m_BackendId(backendId) {}

    ~BackendProfiling()
    {}

    std::unique_ptr<IRegisterBackendCounters>
            GetCounterRegistrationInterface(uint16_t currentMaxGlobalCounterID) override;

    std::unique_ptr<ISendTimelinePacket> GetSendTimelinePacket() override;

    IProfilingGuidGenerator& GetProfilingGuidGenerator() override;

    void ReportCounters(const std::vector<Timestamp>&) override;

    CounterStatus GetCounterStatus(uint16_t backendCounterId) override;

    std::vector<CounterStatus> GetActiveCounters() override;

    bool IsProfilingEnabled() const override;

private:
    IRuntime::CreationOptions m_Options;
    ProfilingService& m_ProfilingService;
    BackendId m_BackendId;
};
}    // namespace profiling
}    // namespace armnn