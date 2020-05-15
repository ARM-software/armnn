//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Types.hpp>
#include <armnn/BackendId.hpp>
#include <armnn/Optional.hpp>

#include <memory>
#include <unordered_map>
#include <functional>

namespace armnn
{

namespace profiling
{
    class ProfilingService;
}
class IBackendInternal;
using IBackendInternalUniquePtr = std::unique_ptr<IBackendInternal>;

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
    void SetProfilingService(armnn::Optional<profiling::ProfilingService&> profilingService);

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

protected:
    using FactoryStorage = std::unordered_map<BackendId, FactoryFunction>;

    /// For testing only
    static void Swap(BackendRegistry& instance, FactoryStorage& other);

private:
    BackendRegistry(const BackendRegistry&) = delete;
    BackendRegistry& operator=(const BackendRegistry&) = delete;

    FactoryStorage m_Factories;
    armnn::Optional<profiling::ProfilingService&> m_ProfilingService;
};

BackendRegistry& BackendRegistryInstance();

} // namespace armnn
