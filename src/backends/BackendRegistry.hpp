//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Types.hpp>
#include <string>
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
    void Register(const std::string& name, FactoryFunction factory);
    FactoryFunction GetFactory(const std::string& name) const;

    struct Helper
    {
        Helper(const std::string& name, FactoryFunction factory)
        {
            BackendRegistry::Instance().Register(name, factory);
        }
    };

    size_t Size() const { return m_BackendFactories.size(); }

protected:
    using FactoryStorage = std::unordered_map<std::string, FactoryFunction>;

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
