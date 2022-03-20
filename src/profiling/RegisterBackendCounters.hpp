//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <client/include/CounterIdMap.hpp>
#include <client/include/IProfilingService.hpp>

#include <client/include/backends/IBackendProfiling.hpp>

#include <common/include/CounterDirectory.hpp>

namespace arm
{

namespace pipe
{

class RegisterBackendCounters : public IRegisterBackendCounters
{
public:

    RegisterBackendCounters(
        uint16_t currentMaxGlobalCounterID, const std::string& backendId, IProfilingService& profilingService)
        : m_CurrentMaxGlobalCounterID(currentMaxGlobalCounterID),
          m_BackendId(backendId),
          m_ProfilingService(profilingService),
          m_CounterDirectory(m_ProfilingService.GetCounterRegistry()) {}

    ~RegisterBackendCounters() = default;

    void RegisterCategory(const std::string& categoryName) override;

    uint16_t RegisterDevice(const std::string& deviceName,
                            uint16_t cores = 0,
                            const arm::pipe::Optional<std::string>& parentCategoryName =
                                arm::pipe::EmptyOptional()) override;

    uint16_t RegisterCounterSet(const std::string& counterSetName,
                                uint16_t count = 0,
                                const arm::pipe::Optional<std::string>& parentCategoryName
                                    = arm::pipe::EmptyOptional()) override;

    uint16_t RegisterCounter(const uint16_t uid,
                             const std::string& parentCategoryName,
                             uint16_t counterClass,
                             uint16_t interpolation,
                             double multiplier,
                             const std::string& name,
                             const std::string& description,
                             const arm::pipe::Optional<std::string>& units      = arm::pipe::EmptyOptional(),
                             const arm::pipe::Optional<uint16_t>& numberOfCores = arm::pipe::EmptyOptional(),
                             const arm::pipe::Optional<uint16_t>& deviceUid     = arm::pipe::EmptyOptional(),
                             const arm::pipe::Optional<uint16_t>& counterSetUid = arm::pipe::EmptyOptional()) override;

private:
    uint16_t m_CurrentMaxGlobalCounterID;
    const std::string& m_BackendId;
    IProfilingService& m_ProfilingService;
    ICounterRegistry& m_CounterDirectory;
};

} // namespace pipe

} // namespace arm
