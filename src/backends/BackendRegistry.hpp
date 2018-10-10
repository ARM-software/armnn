//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Types.hpp>
#include <functional>
#include <memory>
#include <unordered_map>

namespace armnn
{

class IBackend;

class BackendRegistry
{
public:
    using FactoryFunction = std::function<IBackendUniquePtr()>;

    static BackendRegistry& Instance();

    void Register(const BackendId& id, FactoryFunction factory);
    FactoryFunction GetFactory(const BackendId& id) const;

    struct Helper
    {
        Helper(const BackendId& id, FactoryFunction factory)
        {
            BackendRegistry::Instance().Register(id, factory);
        }
    };

    size_t Size() const { return m_BackendFactories.size(); }
    BackendIdSet GetBackendIds() const;

protected:
    using FactoryStorage = std::unordered_map<BackendId, FactoryFunction>;

    // For testing only
    static void Swap(FactoryStorage& other);
    BackendRegistry() {}
    ~BackendRegistry() {}

private:
    BackendRegistry(const BackendRegistry&) = delete;
    BackendRegistry& operator=(const BackendRegistry&) = delete;

    FactoryStorage m_BackendFactories;
};

} // namespace armnn
