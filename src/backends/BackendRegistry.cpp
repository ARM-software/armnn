//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "BackendRegistry.hpp"
#include <armnn/Exceptions.hpp>

namespace armnn
{

BackendRegistry& BackendRegistry::Instance()
{
    static BackendRegistry instance;
    return instance;
}

void BackendRegistry::Register(const std::string& name, FactoryFunction factory)
{
    if (m_BackendFactories.count(name) > 0)
    {
        throw InvalidArgumentException(name + " already registered as backend");
    }

    m_BackendFactories[name] = factory;
}

BackendRegistry::FactoryFunction BackendRegistry::GetFactory(const std::string& name) const
{
    auto it = m_BackendFactories.find(name);
    if (it == m_BackendFactories.end())
    {
        throw InvalidArgumentException(name + " has no backend factory registered");
    }

    return it->second;
}

void BackendRegistry::Swap(BackendRegistry::FactoryStorage& other)
{
    BackendRegistry& instance = Instance();
    std::swap(instance.m_BackendFactories, other);
}

}
