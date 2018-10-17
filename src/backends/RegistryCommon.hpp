//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/BackendId.hpp>
#include <armnn/Exceptions.hpp>
#include <functional>
#include <memory>
#include <unordered_map>

namespace armnn
{

template <typename RegisteredType>
struct RegisteredTypeName
{
    static const char * Name() { return "UNKNOWN"; }
};

template <typename RegisteredType, typename PointerType>
class RegistryCommon
{
public:
    using FactoryFunction = std::function<PointerType()>;

    void Register(const BackendId& id, FactoryFunction factory)
    {
        if (m_Factories.count(id) > 0)
        {
            throw InvalidArgumentException(
                std::string(id) + " already registered as " + RegisteredTypeName<RegisteredType>::Name() + " factory",
                CHECK_LOCATION());
        }

        m_Factories[id] = factory;
    }

    FactoryFunction GetFactory(const BackendId& id) const
    {
        auto it = m_Factories.find(id);
        if (it == m_Factories.end())
        {
            throw InvalidArgumentException(
                std::string(id) + " has no " + RegisteredTypeName<RegisteredType>::Name() + " factory registered",
                CHECK_LOCATION());
        }

        return it->second;
    }

    size_t Size() const
    {
        return m_Factories.size();
    }

    BackendIdSet GetBackendIds() const
    {
        BackendIdSet result;
        for (const auto& it : m_Factories)
        {
            result.insert(it.first);
        }
        return result;
    }

    RegistryCommon() {}
    virtual ~RegistryCommon() {}

protected:
    using FactoryStorage = std::unordered_map<BackendId, FactoryFunction>;

    // For testing only
    static void Swap(RegistryCommon& instance, FactoryStorage& other)
    {
        std::swap(instance.m_Factories, other);
    }

private:
    RegistryCommon(const RegistryCommon&) = delete;
    RegistryCommon& operator=(const RegistryCommon&) = delete;

    FactoryStorage m_Factories;
};

template <typename RegistryType>
struct StaticRegistryInitializer
{
    using FactoryFunction = typename RegistryType::FactoryFunction;

    StaticRegistryInitializer(RegistryType& instance,
                              const BackendId& id,
                              FactoryFunction factory)
    {
        instance.Register(id, factory);
    }
};

} // namespace armnn
