//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/backends/ITensorHandleFactory.hpp>

#include <memory>
#include <vector>

namespace armnn
{

//Forward
class IMemoryManager;

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

    /// Aquire memory required for inference
    void AquireMemory();

    /// Release memory required for inference
    void ReleaseMemory();

private:
    std::vector<std::unique_ptr<ITensorHandleFactory>> m_Factories;
    std::vector<std::shared_ptr<IMemoryManager>> m_MemoryManagers;
};

} // namespace armnn
