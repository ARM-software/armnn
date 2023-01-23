//
// Copyright © 2017-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/backends/IBackendInternal.hpp>

namespace armnn
{

// add new capabilities here..
const BackendCapabilities cpuAccCapabilities("CpuAcc",
                                             {
                                                     {"NonConstWeights", true},
                                                     {"AsyncExecution", false},
                                                     {"ProtectedContentAllocation", false},
                                                     {"ConstantTensorsAsInputs", true},
                                                     {"PreImportIOTensors", false},
                                                     {"ExternallyManagedMemory", true},
                                                     {"MultiAxisPacking", false},
                                                     {"SingleAxisPacking", true}
                                             });


class NeonBackend : public IBackendInternal
{
public:
    NeonBackend()  = default;
    ~NeonBackend() = default;

    static const BackendId& GetIdStatic();
    const BackendId& GetId() const override { return GetIdStatic(); }

    IBackendInternal::IMemoryManagerUniquePtr CreateMemoryManager() const override;

    IWorkloadFactoryPtr CreateWorkloadFactory(
        const IBackendInternal::IMemoryManagerSharedPtr& memoryManager = nullptr) const override;

    IWorkloadFactoryPtr CreateWorkloadFactory(
        class TensorHandleFactoryRegistry& tensorHandleFactoryRegistry) const override;

    IWorkloadFactoryPtr CreateWorkloadFactory( const IMemoryManagerSharedPtr& memoryManager,
                                               const ModelOptions& modelOptions) const override;

    IWorkloadFactoryPtr CreateWorkloadFactory(class TensorHandleFactoryRegistry& tensorHandleFactoryRegistry,
                                              const ModelOptions& modelOptions) const override;

    IBackendInternal::IBackendContextPtr CreateBackendContext(const IRuntime::CreationOptions&) const override;
    IBackendInternal::IBackendProfilingContextPtr CreateBackendProfilingContext(
        const IRuntime::CreationOptions&, IBackendProfilingPtr& backendProfiling) override;
    IBackendInternal::ILayerSupportSharedPtr GetLayerSupport() const override;
    IBackendInternal::ILayerSupportSharedPtr GetLayerSupport(const ModelOptions& modelOptions) const override;

    OptimizationViews OptimizeSubgraphView(const SubgraphView& subgraph,
                                           const ModelOptions& modelOptions) const override;

    std::vector<ITensorHandleFactory::FactoryId> GetHandleFactoryPreferences() const override;

    void RegisterTensorHandleFactories(class TensorHandleFactoryRegistry& registry) override;

    IBackendInternal::IBackendSpecificModelContextPtr CreateBackendSpecificModelContext(
        const ModelOptions& modelOptions) const override;

    BackendCapabilities GetCapabilities() const override
    {
        return cpuAccCapabilities;
    };

    std::unique_ptr<ICustomAllocator> GetDefaultAllocator() const override;
};

} // namespace armnn
