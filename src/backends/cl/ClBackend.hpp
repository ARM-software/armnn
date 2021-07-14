//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/backends/IBackendInternal.hpp>

namespace armnn
{

// add new capabilities here..
const BackendCapabilities gpuAccCapabilities("GpuAcc",
                                             {
                                                     {"NonConstWeights", false},
                                                     {"AsyncExecution", false},
                                                     {"ProtectedContentAllocation", true}
                                             });

class ClBackend : public IBackendInternal
{
public:
    ClBackend() : m_EnableCustomAllocator(false) {};
    ~ClBackend() = default;

    static const BackendId& GetIdStatic();
    const BackendId& GetId() const override { return GetIdStatic(); }

    IBackendInternal::IMemoryManagerUniquePtr CreateMemoryManager() const override;

    IBackendInternal::IWorkloadFactoryPtr CreateWorkloadFactory(
        const IBackendInternal::IMemoryManagerSharedPtr& memoryManager = nullptr) const override;

    IBackendInternal::IWorkloadFactoryPtr CreateWorkloadFactory(
        TensorHandleFactoryRegistry& registry) const override;

    IWorkloadFactoryPtr CreateWorkloadFactory(const IMemoryManagerSharedPtr& memoryManager,
                                              const ModelOptions& modelOptions) const override;

    IWorkloadFactoryPtr CreateWorkloadFactory(class TensorHandleFactoryRegistry& tensorHandleFactoryRegistry,
                                              const ModelOptions& modelOptions) const override;

    IWorkloadFactoryPtr CreateWorkloadFactory(class TensorHandleFactoryRegistry& tensorHandleFactoryRegistry,
                                              const ModelOptions& modelOptions,
                                              MemorySourceFlags inputFlags,
                                              MemorySourceFlags outputFlags) const override;

    std::vector<ITensorHandleFactory::FactoryId> GetHandleFactoryPreferences() const override;

    void RegisterTensorHandleFactories(TensorHandleFactoryRegistry& registry) override;

    void RegisterTensorHandleFactories(TensorHandleFactoryRegistry& registry,
                                       MemorySourceFlags inputFlags,
                                       MemorySourceFlags outputFlags) override;

    IBackendInternal::IBackendContextPtr CreateBackendContext(const IRuntime::CreationOptions&) const override;
    IBackendInternal::IBackendProfilingContextPtr CreateBackendProfilingContext(
        const IRuntime::CreationOptions&, IBackendProfilingPtr& backendProfiling) override;

    IBackendInternal::Optimizations GetOptimizations() const override;
    IBackendInternal::ILayerSupportSharedPtr GetLayerSupport() const override;
    IBackendInternal::ILayerSupportSharedPtr GetLayerSupport(const ModelOptions& modelOptions) const override;

    OptimizationViews OptimizeSubgraphView(const SubgraphView& subgraph,
                                           const ModelOptions& modelOptions) const override;

    IBackendInternal::IBackendSpecificModelContextPtr CreateBackendSpecificModelContext(
        const ModelOptions& modelOptions) const override;

    BackendCapabilities GetCapabilities() const override
    {
        return gpuAccCapabilities;
    };

    virtual bool UseCustomMemoryAllocator(armnn::Optional<std::string&> errMsg) override
    {
        IgnoreUnused(errMsg);

        // Set flag to signal the backend to use a custom memory allocator
        m_EnableCustomAllocator = true;

        return m_EnableCustomAllocator;
    }

    bool m_EnableCustomAllocator;
};

} // namespace armnn
