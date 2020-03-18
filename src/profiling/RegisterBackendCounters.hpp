//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "armnn/backends/profiling/IBackendProfiling.hpp"
#include "CounterIdMap.hpp"
#include "CounterDirectory.hpp"
#include "ProfilingService.hpp"

namespace armnn
{

namespace profiling
{

class RegisterBackendCounters : public IRegisterBackendCounters
{
public:

    RegisterBackendCounters(
        uint16_t currentMaxGlobalCounterID, const BackendId& backendId, ProfilingService& profilingService)
        : m_CurrentMaxGlobalCounterID(currentMaxGlobalCounterID),
          m_BackendId(backendId),
          m_ProfilingService(profilingService),
          m_CounterDirectory(m_ProfilingService.GetCounterRegistry()) {}

    ~RegisterBackendCounters() = default;

    void RegisterCategory(const std::string& categoryName) override;

    uint16_t RegisterDevice(const std::string& deviceName,
                            uint16_t cores = 0,
                            const Optional<std::string>& parentCategoryName = EmptyOptional()) override;

    uint16_t RegisterCounterSet(const std::string& counterSetName,
                                uint16_t count = 0,
                                const Optional<std::string>& parentCategoryName = EmptyOptional()) override;

    uint16_t RegisterCounter(const uint16_t uid,
                             const std::string& parentCategoryName,
                             uint16_t counterClass,
                             uint16_t interpolation,
                             double multiplier,
                             const std::string& name,
                             const std::string& description,
                             const Optional<std::string>& units      = EmptyOptional(),
                             const Optional<uint16_t>& numberOfCores = EmptyOptional(),
                             const Optional<uint16_t>& deviceUid     = EmptyOptional(),
                             const Optional<uint16_t>& counterSetUid = EmptyOptional()) override;

private:
    uint16_t m_CurrentMaxGlobalCounterID;
    const BackendId& m_BackendId;
    ProfilingService& m_ProfilingService;
    ICounterRegistry& m_CounterDirectory;
};

} // namespace profiling

} // namespace armnn