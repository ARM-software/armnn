//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <backendsCommon/TensorHandleFactoryRegistry.hpp>
#include <armnn/backends/IMemoryManager.hpp>

namespace armnn
{

void TensorHandleFactoryRegistry::RegisterFactory(std::unique_ptr <ITensorHandleFactory> newFactory)
{
    if (!newFactory)
    {
        return;
    }

    ITensorHandleFactory::FactoryId id = newFactory->GetId();

    // Don't register duplicates
    for (auto& registeredFactory : m_Factories)
    {
        if (id == registeredFactory->GetId())
        {
            return;
        }
    }

    // Take ownership of the new allocator
    m_Factories.push_back(std::move(newFactory));
}

void TensorHandleFactoryRegistry::RegisterMemoryManager(std::shared_ptr<armnn::IMemoryManager> memoryManger)
{
    m_MemoryManagers.push_back(memoryManger);
}

ITensorHandleFactory* TensorHandleFactoryRegistry::GetFactory(ITensorHandleFactory::FactoryId id) const
{
    for (auto& factory : m_Factories)
    {
        if (factory->GetId() == id)
        {
            return factory.get();
        }
    }

    return nullptr;
}

ITensorHandleFactory* TensorHandleFactoryRegistry::GetFactory(ITensorHandleFactory::FactoryId id,
                                                              MemorySource memSource) const
{
    for (auto& factory : m_Factories)
    {
        if (factory->GetId() == id && factory->GetImportFlags() == static_cast<MemorySourceFlags>(memSource))
        {
            return factory.get();
        }
    }

    return nullptr;
}

void TensorHandleFactoryRegistry::RegisterCopyAndImportFactoryPair(ITensorHandleFactory::FactoryId copyFactoryId,
                                                                   ITensorHandleFactory::FactoryId importFactoryId)
{
    m_FactoryMappings[copyFactoryId] = importFactoryId;
}

ITensorHandleFactory::FactoryId TensorHandleFactoryRegistry::GetMatchingImportFactoryId(
    ITensorHandleFactory::FactoryId copyFactoryId)
{
    return m_FactoryMappings[copyFactoryId];
}

void TensorHandleFactoryRegistry::AquireMemory()
{
    for (auto& mgr : m_MemoryManagers)
    {
        mgr->Acquire();
    }
}

void TensorHandleFactoryRegistry::ReleaseMemory()
{
    for (auto& mgr : m_MemoryManagers)
    {
        mgr->Release();
    }
}

} // namespace armnn
