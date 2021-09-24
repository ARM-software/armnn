//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "MockImportBackend.hpp"
#include "MockImportLayerSupport.hpp"

#include <armnn/BackendRegistry.hpp>
#include <armnn/backends/IBackendContext.hpp>
#include <armnn/backends/IMemoryManager.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>

#include <reference/RefWorkloadFactory.hpp>
#include <reference/RefTensorHandleFactory.hpp>

#include <Optimizer.hpp>

namespace armnn
{

MockImportBackendInitialiser::MockImportBackendInitialiser()
{
    BackendRegistryInstance().Register(MockImportBackend::GetIdStatic(),
                                       []()
                                       {
                                           return IBackendInternalUniquePtr(new MockImportBackend);
                                       });
}

MockImportBackendInitialiser::~MockImportBackendInitialiser()
{
    try
    {
        BackendRegistryInstance().Deregister(MockImportBackend::GetIdStatic());
    }
    catch (...)
    {
        std::cerr << "could not deregister mock import backend" << std::endl;
    }
}

const BackendId& MockImportBackend::GetIdStatic()
{
    static const BackendId s_Id{ MockImportBackendId() };
    return s_Id;
}

IBackendInternal::IWorkloadFactoryPtr MockImportBackend::CreateWorkloadFactory(
    const IBackendInternal::IMemoryManagerSharedPtr& memoryManager) const
{
    return std::make_unique<RefWorkloadFactory>(PolymorphicPointerDowncast<RefMemoryManager>(memoryManager));
}

IBackendInternal::IWorkloadFactoryPtr MockImportBackend::CreateWorkloadFactory(
    class TensorHandleFactoryRegistry& tensorHandleFactoryRegistry) const
{
    auto memoryManager = std::make_shared<RefMemoryManager>();

    tensorHandleFactoryRegistry.RegisterMemoryManager(memoryManager);
    tensorHandleFactoryRegistry.RegisterFactory(std::make_unique<RefTensorHandleFactory>(memoryManager));

    return std::make_unique<RefWorkloadFactory>(PolymorphicPointerDowncast<RefMemoryManager>(memoryManager));
}

IBackendInternal::IBackendContextPtr MockImportBackend::CreateBackendContext(const IRuntime::CreationOptions&) const
{
    return IBackendContextPtr{};
}

IBackendInternal::IBackendProfilingContextPtr MockImportBackend::CreateBackendProfilingContext(
    const IRuntime::CreationOptions&, IBackendProfilingPtr&)
{
    return IBackendProfilingContextPtr{};
}

IBackendInternal::IMemoryManagerUniquePtr MockImportBackend::CreateMemoryManager() const
{
    return std::make_unique<RefMemoryManager>();
}

IBackendInternal::ILayerSupportSharedPtr MockImportBackend::GetLayerSupport() const
{
    static ILayerSupportSharedPtr layerSupport{new MockImportLayerSupport};
    return layerSupport;
}

OptimizationViews MockImportBackend::OptimizeSubgraphView(const SubgraphView& subgraph) const
{
    OptimizationViews optimizationViews;

    optimizationViews.AddUntouchedSubgraph(SubgraphView(subgraph));

    return optimizationViews;
}

std::vector<ITensorHandleFactory::FactoryId> MockImportBackend::GetHandleFactoryPreferences() const
{
    return std::vector<ITensorHandleFactory::FactoryId> { RefTensorHandleFactory::GetIdStatic() };
}

void MockImportBackend::RegisterTensorHandleFactories(class TensorHandleFactoryRegistry& registry)
{
    auto memoryManager = std::make_shared<RefMemoryManager>();

    registry.RegisterMemoryManager(memoryManager);
    registry.RegisterFactory(std::make_unique<RefTensorHandleFactory>(memoryManager));
}

} // namespace armnn
