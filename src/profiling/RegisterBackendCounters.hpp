//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "armnn/backends/profiling/IBackendProfiling.hpp"
#include "CounterIdMap.hpp"
#include "CounterDirectory.hpp"
#include "ProfilingService.hpp"

namespace arm
{

namespace pipe
{

class RegisterBackendCounters : public IRegisterBackendCounters
{
public:

    RegisterBackendCounters(
        uint16_t currentMaxGlobalCounterID, const armnn::BackendId& backendId, ProfilingService& profilingService)
        : m_CurrentMaxGlobalCounterID(currentMaxGlobalCounterID),
          m_BackendId(backendId),
          m_ProfilingService(profilingService),
          m_CounterDirectory(m_ProfilingService.GetCounterRegistry()) {}

    ~RegisterBackendCounters() = default;

    void RegisterCategory(const std::string& categoryName) override;

    uint16_t RegisterDevice(const std::string& deviceName,
                            uint16_t cores = 0,
                            const armnn::Optional<std::string>& parentCategoryName = armnn::EmptyOptional()) override;

    uint16_t RegisterCounterSet(const std::string& counterSetName,
                                uint16_t count = 0,
                                const armnn::Optional<std::string>& parentCategoryName
                                    = armnn::EmptyOptional()) override;

    uint16_t RegisterCounter(const uint16_t uid,
                             const std::string& parentCategoryName,
                             uint16_t counterClass,
                             uint16_t interpolation,
                             double multiplier,
                             const std::string& name,
                             const std::string& description,
                             const armnn::Optional<std::string>& units      = armnn::EmptyOptional(),
                             const armnn::Optional<uint16_t>& numberOfCores = armnn::EmptyOptional(),
                             const armnn::Optional<uint16_t>& deviceUid     = armnn::EmptyOptional(),
                             const armnn::Optional<uint16_t>& counterSetUid = armnn::EmptyOptional()) override;

private:
    uint16_t m_CurrentMaxGlobalCounterID;
    const armnn::BackendId& m_BackendId;
    ProfilingService& m_ProfilingService;
    ICounterRegistry& m_CounterDirectory;
};

} // namespace pipe

} // namespace arm