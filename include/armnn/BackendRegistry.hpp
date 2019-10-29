//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Types.hpp>
#include <armnn/BackendId.hpp>

#include <memory>
#include <unordered_map>
#include <functional>

namespace armnn
{

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

protected:
    using FactoryStorage = std::unordered_map<BackendId, FactoryFunction>;

    // For testing only
    static void Swap(BackendRegistry& instance, FactoryStorage& other);

private:
    BackendRegistry(const BackendRegistry&) = delete;
    BackendRegistry& operator=(const BackendRegistry&) = delete;

    FactoryStorage m_Factories;
};

BackendRegistry& BackendRegistryInstance();

} // namespace armnn
