//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/IRuntime.hpp>
#include <armnn/profiling/IProfilingGuidGenerator.hpp>
#include <armnn/profiling/ISendTimelinePacket.hpp>
#include <memory>
#include <vector>

namespace armnn
{

namespace profiling
{

struct CounterValue
{
    CounterValue(uint16_t id, uint32_t value) :
        counterId(id), counterValue(value) {}
    uint16_t counterId;
    uint32_t counterValue;
};

struct Timestamp
{
    uint64_t timestamp;
    std::vector<CounterValue> counterValues;
};

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

class IRegisterBackendCounters
{
public:
    virtual void RegisterCategory(const std::string& categoryName) = 0;

    virtual uint16_t RegisterDevice(const std::string& deviceName,
                                    uint16_t cores = 0,
                                    const Optional<std::string>& parentCategoryName = EmptyOptional()) = 0;

    virtual uint16_t RegisterCounterSet(const std::string& counterSetName,
                                        uint16_t count = 0,
                                        const Optional<std::string>& parentCategoryName = EmptyOptional()) = 0;

    virtual uint16_t RegisterCounter(const uint16_t uid,
                                     const std::string& parentCategoryName,
                                     uint16_t counterClass,
                                     uint16_t interpolation,
                                     double multiplier,
                                     const std::string& name,
                                     const std::string& description,
                                     const Optional<std::string>& units      = EmptyOptional(),
                                     const Optional<uint16_t>& numberOfCores = EmptyOptional(),
                                     const Optional<uint16_t>& deviceUid     = EmptyOptional(),
                                     const Optional<uint16_t>& counterSetUid = EmptyOptional()) = 0;

    virtual ~IRegisterBackendCounters() {}
};

class IBackendProfiling
{
public:
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

}    // namespace profiling

}    // namespace armnn