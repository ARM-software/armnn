//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TosaRefBackend.hpp"
#include "TosaRefBackendId.hpp"
#include "TosaRefWorkloadFactory.hpp"
#include "TosaRefLayerSupport.hpp"
#include "TosaRefTensorHandleFactory.hpp"

#include <armnn/BackendRegistry.hpp>
#include <armnn/backends/IBackendContext.hpp>
#include <armnn/backends/IMemoryManager.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>
#include <backendsCommon/DefaultAllocator.hpp>
#include <backendsCommon/SubgraphUtils.hpp>

#include <Optimizer.hpp>

namespace armnn
{

const BackendId& TosaRefBackend::GetIdStatic()
{
    static const BackendId s_Id{TosaRefBackendId()};
    return s_Id;
}

IBackendInternal::IWorkloadFactoryPtr TosaRefBackend::CreateWorkloadFactory(
    const IBackendInternal::IMemoryManagerSharedPtr& memoryManager) const
{
    return std::make_unique<TosaRefWorkloadFactory>(PolymorphicPointerDowncast<TosaRefMemoryManager>(memoryManager));
}

IBackendInternal::IWorkloadFactoryPtr TosaRefBackend::CreateWorkloadFactory(
    class TensorHandleFactoryRegistry& tensorHandleFactoryRegistry) const
{
    auto memoryManager = std::make_shared<TosaRefMemoryManager>();

    tensorHandleFactoryRegistry.RegisterMemoryManager(memoryManager);

    auto factory = std::make_unique<TosaRefTensorHandleFactory>(memoryManager);
    // Register copy and import factory pair
    tensorHandleFactoryRegistry.RegisterCopyAndImportFactoryPair(factory->GetId(), factory->GetId());
    // Register the factory
    tensorHandleFactoryRegistry.RegisterFactory(std::move(factory));

    return std::make_unique<TosaRefWorkloadFactory>(PolymorphicPointerDowncast<TosaRefMemoryManager>(memoryManager));
}

IBackendInternal::IBackendContextPtr TosaRefBackend::CreateBackendContext(const IRuntime::CreationOptions&) const
{
    return IBackendContextPtr{};
}

IBackendInternal::IBackendProfilingContextPtr TosaRefBackend::CreateBackendProfilingContext(
    const IRuntime::CreationOptions&, IBackendProfilingPtr&)
{
    return IBackendProfilingContextPtr{};
}

IBackendInternal::IMemoryManagerUniquePtr TosaRefBackend::CreateMemoryManager() const
{
    return std::make_unique<TosaRefMemoryManager>();
}

IBackendInternal::ILayerSupportSharedPtr TosaRefBackend::GetLayerSupport() const
{
    static ILayerSupportSharedPtr layerSupport{new TosaRefLayerSupport};
    return layerSupport;
}

OptimizationViews TosaRefBackend::OptimizeSubgraphView(const SubgraphView& subgraph,
                                                       const ModelOptions& modelOptions) const
{
    OptimizationViews optimizationViews(modelOptions);
    optimizationViews.AddUntouchedSubgraph(SubgraphView(subgraph));

    return optimizationViews;
}

std::vector<ITensorHandleFactory::FactoryId> TosaRefBackend::GetHandleFactoryPreferences() const
{
    return std::vector<ITensorHandleFactory::FactoryId> { TosaRefTensorHandleFactory::GetIdStatic() };
}

void TosaRefBackend::RegisterTensorHandleFactories(class TensorHandleFactoryRegistry& registry)
{
    auto memoryManager = std::make_shared<TosaRefMemoryManager>();

    registry.RegisterMemoryManager(memoryManager);

    auto factory = std::make_unique<TosaRefTensorHandleFactory>(memoryManager);

    // Register copy and import factory pair
    registry.RegisterCopyAndImportFactoryPair(factory->GetId(), factory->GetId());
    // Register the factory
    registry.RegisterFactory(std::move(factory));
}

std::unique_ptr<ICustomAllocator> TosaRefBackend::GetDefaultAllocator() const
{
    return std::make_unique<DefaultAllocator>();
}

} // namespace armnn
