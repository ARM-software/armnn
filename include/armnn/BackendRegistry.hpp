//
// Copyright © 2017,2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/BackendId.hpp>
#include <armnn/Optional.hpp>
#include <memory>
#include <unordered_map>
#include <functional>
#include <stddef.h>
#include <string>

namespace arm
{
namespace pipe
{

class IProfilingService;

} // namespace arm
} // namespace pipe

namespace armnn
{

class IBackendInternal;
class ICustomAllocator;
class IMemoryOptimizerStrategy;

using IBackendInternalUniquePtr = std::unique_ptr<IBackendInternal>;
using MemoryOptimizerStrategiesMapRef = std::unordered_map<BackendId, std::shared_ptr<IMemoryOptimizerStrategy>>;

class BackendRegistry
{
public:
    using PointerType = IBackendInternalUniquePtr;
    using FactoryFunction = std::function<PointerType()>;

    void Register(const BackendId& id, FactoryFunction factory);
    bool IsBackendRegistered(const BackendId& id) const;
    FactoryFunction GetFactory(const BackendId& id) const;
    size_t Size() const;
    BackendIdSet GetBackendIds() const;
    std::string GetBackendIdsAsString() const;
    void SetProfilingService(armnn::Optional<arm::pipe::IProfilingService&> profilingService);
    void RegisterAllocator(const BackendId& id, std::shared_ptr<ICustomAllocator> alloc);
    std::unordered_map<BackendId, std::shared_ptr<ICustomAllocator>> GetAllocators();
    void RegisterMemoryOptimizerStrategy(const BackendId& id, std::shared_ptr<IMemoryOptimizerStrategy> strategy);
    MemoryOptimizerStrategiesMapRef GetMemoryOptimizerStrategies();

    BackendRegistry() {}
    virtual ~BackendRegistry() {}

    struct StaticRegistryInitializer
    {
        StaticRegistryInitializer(BackendRegistry& instance,
                                  const BackendId& id,
                                  FactoryFunction factory)
        {
            instance.Register(id, factory);
        }
    };

    void Deregister(const BackendId& id);
    void DeregisterAllocator(const BackendId &id);
    void DeregisterMemoryOptimizerStrategy(const BackendId &id);

protected:
    using FactoryStorage = std::unordered_map<BackendId, FactoryFunction>;

    /// For testing only
    static void Swap(BackendRegistry& instance, FactoryStorage& other);

private:
    BackendRegistry(const BackendRegistry&) = delete;
    BackendRegistry& operator=(const BackendRegistry&) = delete;

    FactoryStorage m_Factories;
    armnn::Optional<arm::pipe::IProfilingService&> m_ProfilingService;
    std::unordered_map<BackendId, std::shared_ptr<ICustomAllocator>> m_CustomMemoryAllocatorMap;
    std::unordered_map<BackendId, std::shared_ptr<IMemoryOptimizerStrategy>> m_MemoryOptimizerStrategyMap;
};

BackendRegistry& BackendRegistryInstance();

} // namespace armnn
