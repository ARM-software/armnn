//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <client/include/CounterStatus.hpp>
#include <client/include/CounterValue.hpp>
#include <client/include/IProfilingService.hpp>
#include <client/include/ISendCounterPacket.hpp>
#include <client/include/ISendTimelinePacket.hpp>
#include <client/include/ProfilingOptions.hpp>
#include <client/include/Timestamp.hpp>

#include <common/include/IProfilingGuidGenerator.hpp>
#include <common/include/Optional.hpp>

#include <memory>
#include <vector>

namespace arm
{

namespace pipe
{

class IRegisterBackendCounters
{
public:
    virtual void RegisterCategory(const std::string& categoryName) = 0;

    virtual uint16_t RegisterDevice(const std::string& deviceName,
                                    uint16_t cores = 0,
                                    const arm::pipe::Optional<std::string>& parentCategoryName
                                        = arm::pipe::EmptyOptional()) = 0;

    virtual uint16_t RegisterCounterSet(const std::string& counterSetName,
                                        uint16_t count = 0,
                                        const arm::pipe::Optional<std::string>& parentCategoryName
                                            = arm::pipe::EmptyOptional()) = 0;

    virtual uint16_t RegisterCounter(const uint16_t uid,
        const std::string& parentCategoryName,
        uint16_t counterClass,
        uint16_t interpolation,
        double multiplier,
        const std::string& name,
        const std::string& description,
        const arm::pipe::Optional<std::string>& units      = arm::pipe::EmptyOptional(),
        const arm::pipe::Optional<uint16_t>& numberOfCores = arm::pipe::EmptyOptional(),
        const arm::pipe::Optional<uint16_t>& deviceUid     = arm::pipe::EmptyOptional(),
        const arm::pipe::Optional<uint16_t>& counterSetUid = arm::pipe::EmptyOptional()) = 0;

    virtual ~IRegisterBackendCounters() {}
};

class IBackendProfiling
{
public:
    static std::unique_ptr<IBackendProfiling> CreateBackendProfiling(const ProfilingOptions& options,
                                                                     IProfilingService& profilingService,
                                                                     const std::string& backendId);
    virtual ~IBackendProfiling()
    {}

    virtual std::unique_ptr<IRegisterBackendCounters>
            GetCounterRegistrationInterface(uint16_t currentMaxGlobalCounterID) = 0;

    virtual std::unique_ptr<ISendTimelinePacket> GetSendTimelinePacket() = 0;

    virtual IProfilingGuidGenerator& GetProfilingGuidGenerator() = 0;

    virtual void ReportCounters(const std::vector<Timestamp>& counterValues) = 0;

    virtual CounterStatus GetCounterStatus(uint16_t backendCounterId) = 0;

    virtual std::vector<CounterStatus> GetActiveCounters() = 0;

    virtual bool IsProfilingEnabled() const = 0;

};

}    // namespace pipe

}    // namespace arm
