//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SampleDynamicBackend.hpp"
#include "SampleDynamicLayerSupport.hpp"
#include "SampleDynamicWorkloadFactory.hpp"
#include "SampleMemoryManager.hpp"
#include "SampleDynamicTensorHandleFactory.hpp"

#include <armnn/backends/IBackendInternal.hpp>
#include <armnn/backends/OptimizationViews.hpp>

namespace sdb // sample dynamic backend
{

constexpr const char * SampleDynamicBackendId() { return "SampleDynamic"; }

class SampleDynamicBackend : public armnn::IBackendInternal
{
public:
    SampleDynamicBackend()  = default;
    ~SampleDynamicBackend() = default;

    static const armnn::BackendId& GetIdStatic()
    {
        static const armnn::BackendId s_Id{SampleDynamicBackendId()};
        return s_Id;
    }

    const armnn::BackendId& GetId() const override { return GetIdStatic(); }

    IBackendInternal::IMemoryManagerUniquePtr CreateMemoryManager() const override
    {
        return std::make_unique<SampleMemoryManager>();
    }

    IBackendInternal::IWorkloadFactoryPtr CreateWorkloadFactory(
        const IMemoryManagerSharedPtr& memoryManager) const override
    {
        return std::make_unique<SampleDynamicWorkloadFactory>(
                armnn::PolymorphicPointerDowncast<SampleMemoryManager>(memoryManager));
    }

    IBackendInternal::IWorkloadFactoryPtr CreateWorkloadFactory(
        class armnn::TensorHandleFactoryRegistry& tensorHandleFactoryRegistry) const override
    {
        auto memoryManager = std::make_shared<SampleMemoryManager>();

        tensorHandleFactoryRegistry.RegisterMemoryManager(memoryManager);
        tensorHandleFactoryRegistry.RegisterFactory(std::make_unique<SampleDynamicTensorHandleFactory>(memoryManager));

        return std::make_unique<SampleDynamicWorkloadFactory>(
                armnn::PolymorphicPointerDowncast<SampleMemoryManager>(memoryManager));
    }

    IBackendInternal::IBackendProfilingContextPtr CreateBackendProfilingContext(
        const armnn::IRuntime::CreationOptions&, IBackendProfilingPtr&) override
    {
        return IBackendProfilingContextPtr{};
    }

    IBackendInternal::ILayerSupportSharedPtr GetLayerSupport() const override
    {
        static ILayerSupportSharedPtr layerSupport{new SampleDynamicLayerSupport};
        return layerSupport;
    }

    std::vector<armnn::ITensorHandleFactory::FactoryId> GetHandleFactoryPreferences() const override
    {
        return std::vector<armnn::ITensorHandleFactory::FactoryId> { SampleDynamicTensorHandleFactory::GetIdStatic() };
    }

    IBackendInternal::IBackendContextPtr CreateBackendContext(const armnn::IRuntime::CreationOptions&) const override
    {
        return IBackendContextPtr{};
    }

    armnn::OptimizationViews OptimizeSubgraphView(const armnn::SubgraphView& subgraph) const override
    {
        armnn::OptimizationViews optimizationViews;

        optimizationViews.AddUntouchedSubgraph(armnn::SubgraphView(subgraph));

        return optimizationViews;
    }

    void RegisterTensorHandleFactories(class armnn::TensorHandleFactoryRegistry& registry) override
    {
        auto memoryManager = std::make_shared<SampleMemoryManager>();

        registry.RegisterMemoryManager(memoryManager);
        registry.RegisterFactory(std::make_unique<SampleDynamicTensorHandleFactory>(memoryManager));
    }

};

} // namespace sdb

const char* GetBackendId()
{
    return sdb::SampleDynamicBackend::GetIdStatic().Get().c_str();
}

void GetVersion(uint32_t* outMajor, uint32_t* outMinor)
{
    if (!outMajor || !outMinor)
    {
        return;
    }

    armnn::BackendVersion apiVersion = armnn::IBackendInternal::GetApiVersion();

    *outMajor = apiVersion.m_Major;
    *outMinor = apiVersion.m_Minor;
}

void* BackendFactory()
{
    return new sdb::SampleDynamicBackend();
}


