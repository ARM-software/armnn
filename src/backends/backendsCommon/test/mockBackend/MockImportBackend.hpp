//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/backends/IBackendInternal.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>

namespace armnn
{

constexpr const char* MockImportBackendId() { return "MockRef"; }

class MockImportBackendInitialiser
{
public:
    MockImportBackendInitialiser();
    ~MockImportBackendInitialiser();
};

class MockImportBackend : public IBackendInternal
{
public:
    MockImportBackend()  = default;
    ~MockImportBackend() = default;

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

    OptimizationViews OptimizeSubgraphView(const SubgraphView& subgraph) const override;

    std::vector<ITensorHandleFactory::FactoryId> GetHandleFactoryPreferences() const override;

    void RegisterTensorHandleFactories(class TensorHandleFactoryRegistry& registry) override;
};

} // namespace armnn
