//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <client/include/backends/IBackendProfiling.hpp>
#include <client/include/IProfilingService.hpp>

namespace arm
{

namespace pipe
{

class BackendProfiling : public IBackendProfiling
{
public:
    BackendProfiling(const ProfilingOptions& options,
                     IProfilingService& profilingService,
                     const std::string& backendId)
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
    ProfilingOptions m_Options;
    IProfilingService& m_ProfilingService;
    std::string m_BackendId;
};

}    // namespace pipe

}    // namespace arm
