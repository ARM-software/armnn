//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/BackendRegistry.hpp>
#include <armnn/Exceptions.hpp>
#include <ProfilingService.hpp>

namespace armnn
{

BackendRegistry& BackendRegistryInstance()
{
    static BackendRegistry instance;
    return instance;
}

void BackendRegistry::Register(const BackendId& id, BackendRegistry::FactoryFunction factory)
{
    if (m_Factories.find(id) != m_Factories.end())
    {
        throw InvalidArgumentException(
            std::string(id) + " already registered as IBackend factory",
            CHECK_LOCATION());
    }
    m_Factories[id] = factory;

    if (m_ProfilingService.has_value())
    {
        if (m_ProfilingService.has_value() && m_ProfilingService.value().IsProfilingEnabled())
        {
            m_ProfilingService.value().IncrementCounterValue(armnn::profiling::REGISTERED_BACKENDS);
        }
    }

}

void BackendRegistry::Deregister(const BackendId& id)
{
    m_Factories.erase(id);
    DeregisterAllocator(id);

    if (m_ProfilingService.has_value() && m_ProfilingService.value().IsProfilingEnabled())
    {
        m_ProfilingService.value().IncrementCounterValue(armnn::profiling::UNREGISTERED_BACKENDS);
    }
}

bool BackendRegistry::IsBackendRegistered(const BackendId& id) const
{
    return (m_Factories.find(id) != m_Factories.end());
}

BackendRegistry::FactoryFunction BackendRegistry::GetFactory(const BackendId& id) const
{
    auto it = m_Factories.find(id);
    if (it == m_Factories.end())
    {
        throw InvalidArgumentException(
            std::string(id) + " has no IBackend factory registered",
            CHECK_LOCATION());
    }

    return it->second;
}

size_t BackendRegistry::Size() const
{
    return m_Factories.size();
}

BackendIdSet BackendRegistry::GetBackendIds() const
{
    BackendIdSet result;
    for (const auto& it : m_Factories)
    {
        result.insert(it.first);
    }
    return result;
}

std::string BackendRegistry::GetBackendIdsAsString() const
{
    static const std::string delimitator = ", ";

    std::stringstream output;
    for (auto& backendId : GetBackendIds())
    {
        if (output.tellp() != std::streampos(0))
        {
            output << delimitator;
        }
        output << backendId;
    }

    return output.str();
}

void BackendRegistry::Swap(BackendRegistry& instance, BackendRegistry::FactoryStorage& other)
{
    std::swap(instance.m_Factories, other);
}

void BackendRegistry::SetProfilingService(armnn::Optional<profiling::ProfilingService&> profilingService)
{
    m_ProfilingService = profilingService;
}

void BackendRegistry::RegisterAllocator(const BackendId& id, std::shared_ptr<ICustomAllocator> alloc)
{
    if (m_CustomMemoryAllocatorMap.find(id) != m_CustomMemoryAllocatorMap.end())
    {
        throw InvalidArgumentException(
            std::string(id) + " already has an allocator associated with it",
            CHECK_LOCATION());
    }
    m_CustomMemoryAllocatorMap[id] = alloc;
}

void BackendRegistry::DeregisterAllocator(const BackendId& id)
{
    m_CustomMemoryAllocatorMap.erase(id);
}

std::unordered_map<BackendId, std::shared_ptr<ICustomAllocator>> BackendRegistry::GetAllocators()
{
    return m_CustomMemoryAllocatorMap;
}

void BackendRegistry::RegisterMemoryOptimizerStrategy(const BackendId& id,
                                                      std::shared_ptr<IMemoryOptimizerStrategy> strategy)
{
    if (m_MemoryOptimizerStrategyMap.find(id) != m_MemoryOptimizerStrategyMap.end())
    {
        throw InvalidArgumentException(
            std::string(id) + " already has an memory optimizer strategy associated with it",
            CHECK_LOCATION());
    }
    m_MemoryOptimizerStrategyMap[id] = strategy;
}

void BackendRegistry::DeregisterMemoryOptimizerStrategy(const BackendId &id)
{
    m_MemoryOptimizerStrategyMap.erase(id);
}

MemoryOptimizerStrategiesMapRef BackendRegistry::GetMemoryOptimizerStrategies()
{
    return m_MemoryOptimizerStrategyMap;
}

} // namespace armnn
