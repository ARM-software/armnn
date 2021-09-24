//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/BackendOptions.hpp>
#include <armnn/backends/IBackendInternal.hpp>

namespace armnn
{

IMemoryManagerUniquePtr IBackendInternal::CreateMemoryManager() const
{
    return IMemoryManagerUniquePtr();
}

IBackendInternal::IWorkloadFactoryPtr IBackendInternal::CreateWorkloadFactory(
    class TensorHandleFactoryRegistry& /*tensorHandleFactoryRegistry*/) const
{
    return IWorkloadFactoryPtr{};
}

IBackendInternal::IWorkloadFactoryPtr IBackendInternal::CreateWorkloadFactory(
    const IMemoryManagerSharedPtr& memoryManager,
    const ModelOptions& modelOptions) const
{
    if (!modelOptions.empty())
    {
        for (auto optionsGroup : modelOptions)
        {
            if (optionsGroup.GetBackendId() == GetId())
            {
                return IWorkloadFactoryPtr{};
            }
        }
    }

    return CreateWorkloadFactory(memoryManager);
}

IBackendInternal::IWorkloadFactoryPtr IBackendInternal::CreateWorkloadFactory(
    class TensorHandleFactoryRegistry& tensorHandleFactoryRegistry,
    const ModelOptions& modelOptions) const
{
    if (!modelOptions.empty())
    {
        for (auto optionsGroup : modelOptions)
        {
            if (optionsGroup.GetBackendId() == GetId())
            {
                return IWorkloadFactoryPtr{};
            }
        }
    }

    return CreateWorkloadFactory(tensorHandleFactoryRegistry);
}

IBackendInternal::IWorkloadFactoryPtr IBackendInternal::CreateWorkloadFactory(
    class TensorHandleFactoryRegistry& tensorHandleFactoryRegistry,
    const ModelOptions& modelOptions,
    MemorySourceFlags inputFlags,
    MemorySourceFlags outputFlags) const
{
    IgnoreUnused(inputFlags);
    IgnoreUnused(outputFlags);
    return CreateWorkloadFactory(tensorHandleFactoryRegistry, modelOptions);
}

IBackendInternal::IBackendContextPtr IBackendInternal::CreateBackendContext(const IRuntime::CreationOptions&) const
{
    return IBackendContextPtr{};
}

IBackendInternal::IBackendSpecificModelContextPtr IBackendInternal::CreateBackendSpecificModelContext(
    const ModelOptions&) const
{
    return IBackendSpecificModelContextPtr{};
}

IBackendInternal::IBackendProfilingContextPtr IBackendInternal::CreateBackendProfilingContext(
    const IRuntime::CreationOptions&, IBackendProfilingPtr&)
{
    return IBackendProfilingContextPtr{};
}

IBackendInternal::ILayerSupportSharedPtr IBackendInternal::GetLayerSupport(const ModelOptions& modelOptions) const
{
    if (!modelOptions.empty())
    {
        for (auto optionsGroup : modelOptions)
        {
            if (optionsGroup.GetBackendId() == GetId())
            {
                return ILayerSupportSharedPtr{};
            }
        }
    }

    return GetLayerSupport();
}

// Default implementation of OptimizeSubgraphView. Returns an untouched subgraph.
// Override this method with a custom optimization implementation.
OptimizationViews IBackendInternal::OptimizeSubgraphView(const SubgraphView& subgraph) const
{
    OptimizationViews result;
    result.AddUntouchedSubgraph(SubgraphView(subgraph));

    return result;
}

OptimizationViews IBackendInternal::OptimizeSubgraphView(const SubgraphView& subgraph,
                                                         const ModelOptions& /*modelOptions*/) const
{
    return OptimizeSubgraphView(subgraph);
}

bool IBackendInternal::SupportsTensorAllocatorAPI() const
{
    return !GetHandleFactoryPreferences().empty();
}

void IBackendInternal::RegisterTensorHandleFactories(class TensorHandleFactoryRegistry& registry,
                                                     MemorySourceFlags /*inputFlags*/,
                                                     MemorySourceFlags /*outputFlags*/)
{
    return RegisterTensorHandleFactories(registry);
}

ITensorHandleFactory::FactoryId IBackendInternal::GetBackwardCompatibleFavoriteHandleFactory()
{
    auto favorites = GetHandleFactoryPreferences();
    if (favorites.empty())
    {
        return ITensorHandleFactory::LegacyFactoryId;
    }

    return favorites[0];
}

std::vector<ITensorHandleFactory::FactoryId> IBackendInternal::GetHandleFactoryPreferences() const
{
    return std::vector<ITensorHandleFactory::FactoryId>();
}

} // namespace armnn
