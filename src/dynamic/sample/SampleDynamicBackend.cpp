//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SampleDynamicBackend.hpp"
#include "SampleDynamicLayerSupport.hpp"
#include "SampleDynamicWorkloadFactory.hpp"
#include "SampleMemoryManager.hpp"

#include <armnn/backends/IBackendInternal.hpp>
#include <armnn/backends/OptimizationViews.hpp>

namespace armnn
{

constexpr const char * SampleDynamicBackendId() { return "SampleDynamic"; }

class SampleDynamicBackend : public IBackendInternal
{
public:
    SampleDynamicBackend()  = default;
    ~SampleDynamicBackend() = default;

    static const BackendId& GetIdStatic()
    {
        static const BackendId s_Id{SampleDynamicBackendId()};
        return s_Id;
    }

    const BackendId& GetId() const override { return GetIdStatic(); }

    IBackendInternal::IMemoryManagerUniquePtr CreateMemoryManager() const override
    {
        return std::make_unique<SampleMemoryManager>();
    }

    IBackendInternal::IWorkloadFactoryPtr CreateWorkloadFactory(
        const IMemoryManagerSharedPtr& memoryManager) const override
    {
        return std::make_unique<SampleDynamicWorkloadFactory>();
    }

    IBackendInternal::IWorkloadFactoryPtr CreateWorkloadFactory(
        class TensorHandleFactoryRegistry& /*tensorHandleFactoryRegistry*/) const override
    {
        return IWorkloadFactoryPtr{};
    }

    IBackendInternal::IBackendProfilingContextPtr CreateBackendProfilingContext(
        const IRuntime::CreationOptions&, IBackendProfilingPtr&) override
    {
        return IBackendProfilingContextPtr{};
    }

    IBackendInternal::ILayerSupportSharedPtr GetLayerSupport() const override
    {
        static ILayerSupportSharedPtr layerSupport{new SampleDynamicLayerSupport};
        return layerSupport;
    }

    std::vector<ITensorHandleFactory::FactoryId> GetHandleFactoryPreferences() const override
    {
        return std::vector<ITensorHandleFactory::FactoryId>();
    }

    IBackendInternal::IBackendContextPtr CreateBackendContext(const IRuntime::CreationOptions&) const override
    {
        return IBackendContextPtr{};
    }

    OptimizationViews OptimizeSubgraphView(const SubgraphView& subgraph) const override
    {
        OptimizationViews optimizationViews;

        optimizationViews.AddUntouchedSubgraph(SubgraphView(subgraph));

        return optimizationViews;
    }
};

} // namespace armnn

const char* GetBackendId()
{
    return armnn::SampleDynamicBackend::GetIdStatic().Get().c_str();
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
    return new armnn::SampleDynamicBackend();
}


