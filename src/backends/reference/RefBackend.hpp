//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/backends/IBackendInternal.hpp>

namespace armnn
{
// add new capabilities here..
const BackendCapabilities cpuRefCapabilities("CpuRef",
                                             {
                                                    {"NonConstWeights", true},
                                                    {"AsyncExecution", true},
                                                    {"ProtectedContentAllocation", false},
                                                    {"ConstantTensorsAsInputs", true},
                                                    {"PreImportIOTensors", true},
                                                    {"ExternallyManagedMemory", true},
                                                    {"MultiAxisPacking", false},
                                                    {"SingleAxisPacking", true}
                                             });

const std::set<armnn::BackendCapability> oldCpuRefCapabilities {
        armnn::BackendCapability::NonConstWeights,
};


class RefBackend : public IBackendInternal
{
public:
    RefBackend()  = default;
    ~RefBackend() = default;

    static const BackendId& GetIdStatic();
    const BackendId& GetId() const override { return GetIdStatic(); }

    IBackendInternal::IMemoryManagerUniquePtr CreateMemoryManager() const override;

    IBackendInternal::IWorkloadFactoryPtr CreateWorkloadFactory(
        const IBackendInternal::IMemoryManagerSharedPtr& memoryManager = nullptr) const override;

    IBackendInternal::IWorkloadFactoryPtr CreateWorkloadFactory(
        class TensorHandleFactoryRegistry& tensorHandleFactoryRegistry) const override;

    IBackendInternal::IBackendContextPtr CreateBackendContext(const IRuntime::CreationOptions&) const override;

    IBackendInternal::IBackendProfilingContextPtr CreateBackendProfilingContext(
        const IRuntime::CreationOptions& creationOptions, IBackendProfilingPtr& backendProfiling) override;

    IBackendInternal::ILayerSupportSharedPtr GetLayerSupport() const override;

    OptimizationViews OptimizeSubgraphView(const SubgraphView& subgraph,
                                           const ModelOptions& modelOptions) const override;

    std::vector<ITensorHandleFactory::FactoryId> GetHandleFactoryPreferences() const override;

    void RegisterTensorHandleFactories(class TensorHandleFactoryRegistry& registry) override;

    BackendCapabilities GetCapabilities() const override
    {
        return cpuRefCapabilities;
    };

    std::unique_ptr<ICustomAllocator> GetDefaultAllocator() const override;

    ExecutionData CreateExecutionData(WorkingMemDescriptor& workingMemDescriptor) const override;

    void UpdateExecutionData(ExecutionData& executionData, WorkingMemDescriptor& workingMemDescriptor) const override;
};

} // namespace armnn
