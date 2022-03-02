//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "CounterIdMap.hpp"
#include "Holder.hpp"
#include "ICounterValues.hpp"
#include "ICounterRegistry.hpp"
#include "IProfilingServiceStatus.hpp"
#include "ISendCounterPacket.hpp"
#include "IReportStructure.hpp"
#include "ProfilingState.hpp"

#include <armnn/backends/profiling/IBackendProfilingContext.hpp>
#include <armnn/profiling/ProfilingOptions.hpp>
#include <armnn/Optional.hpp>
#include <common/include/ProfilingGuidGenerator.hpp>

namespace arm
{

namespace pipe
{

class IProfilingService : public IProfilingGuidGenerator,
                          public IProfilingServiceStatus,
                          public IReadWriteCounterValues
{
public:
    static std::unique_ptr<IProfilingService> CreateProfilingService(
        armnn::Optional<IReportStructure&> reportStructure = armnn::EmptyOptional());
    virtual ~IProfilingService() {};
    virtual std::unique_ptr<ISendTimelinePacket> GetSendTimelinePacket() const = 0;
    virtual const ICounterMappings& GetCounterMappings() const = 0;
    virtual ISendCounterPacket& GetSendCounterPacket() = 0;
    virtual bool IsProfilingEnabled() const = 0;
    virtual bool IsTimelineReportingEnabled() const = 0;
    virtual CaptureData GetCaptureData() = 0;
    virtual ProfilingState GetCurrentState() const = 0;
    // Resets the profiling options, optionally clears the profiling service entirely
    virtual void ResetExternalProfilingOptions(const ProfilingOptions& options,
                                               bool resetProfilingService = false) = 0;
    virtual ProfilingState ConfigureProfilingService(const ProfilingOptions& options,
                                                     bool resetProfilingService = false) = 0;
    // Store a profiling context returned from a backend that support profiling.
    virtual void AddBackendProfilingContext(const armnn::BackendId backendId,
        std::shared_ptr<IBackendProfilingContext> profilingContext) = 0;
    virtual ICounterRegistry& GetCounterRegistry() = 0;
    virtual IRegisterCounterMapping& GetCounterMappingRegistry() = 0;
    // IProfilingGuidGenerator functions
    /// Return the next random Guid in the sequence
    ProfilingDynamicGuid NextGuid() override;
    /// Create a ProfilingStaticGuid based on a hash of the string
    ProfilingStaticGuid GenerateStaticId(const std::string& str) override;
    static ProfilingDynamicGuid GetNextGuid();
    static ProfilingStaticGuid GetStaticId(const std::string& str);
    void ResetGuidGenerator();

private:
    static ProfilingGuidGenerator m_GuidGenerator;
};

} // namespace pipe

} // namespace arm
