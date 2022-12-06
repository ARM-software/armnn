//
// Copyright © 2017,2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Types.hpp>
#include <armnn/backends/ITensorHandleFactory.hpp>
#include <map>
#include <memory>
#include <vector>

namespace armnn
{

//Forward
class IMemoryManager;

using CopyAndImportFactoryPairs = std::map<ITensorHandleFactory::FactoryId, ITensorHandleFactory::FactoryId>;

///
class TensorHandleFactoryRegistry
{
public:
    TensorHandleFactoryRegistry() = default;

    TensorHandleFactoryRegistry(const TensorHandleFactoryRegistry& other) = delete;
    TensorHandleFactoryRegistry(TensorHandleFactoryRegistry&& other) = delete;

    /// Register a TensorHandleFactory and transfer ownership
    void RegisterFactory(std::unique_ptr<ITensorHandleFactory> allocator);

    /// Register a memory manager with shared ownership
    void RegisterMemoryManager(std::shared_ptr<IMemoryManager> memoryManger);

    /// Find a TensorHandleFactory by Id
    /// Returns nullptr if not found
    ITensorHandleFactory* GetFactory(ITensorHandleFactory::FactoryId id) const;

    /// Overload of above allowing specification of Memory Source
    ITensorHandleFactory* GetFactory(ITensorHandleFactory::FactoryId id,
                                     MemorySource memSource) const;

    /// Register a pair of TensorHandleFactory Id for Memory Copy and TensorHandleFactory Id for Memory Import
    void RegisterCopyAndImportFactoryPair(ITensorHandleFactory::FactoryId copyFactoryId,
                                          ITensorHandleFactory::FactoryId importFactoryId);

    /// Get a matching TensorHandleFatory Id for Memory Import given TensorHandleFactory Id for Memory Copy
    ITensorHandleFactory::FactoryId GetMatchingImportFactoryId(ITensorHandleFactory::FactoryId copyFactoryId);

    /// Aquire memory required for inference
    void AquireMemory();

    /// Release memory required for inference
    void ReleaseMemory();

    std::vector<std::shared_ptr<IMemoryManager>>& GetMemoryManagers()
    {
        return m_MemoryManagers;
    }

private:
    std::vector<std::unique_ptr<ITensorHandleFactory>> m_Factories;
    std::vector<std::shared_ptr<IMemoryManager>> m_MemoryManagers;
    CopyAndImportFactoryPairs m_FactoryMappings;
};

} // namespace armnn
