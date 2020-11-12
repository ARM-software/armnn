//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/BackendOptions.hpp>
#include <armnn/backends/IBackendInternal.hpp>

namespace armnn
{

ARMNN_NO_DEPRECATE_WARN_BEGIN
IBackendInternal::ISubGraphConverterPtr IBackendInternal::CreateSubGraphConverter(
    const std::shared_ptr<SubGraph>& /*subGrapg*/) const
{
    return ISubGraphConverterPtr{};
}

IBackendInternal::Optimizations IBackendInternal::GetOptimizations() const
{
    return Optimizations{};
}

IBackendInternal::SubGraphUniquePtr IBackendInternal::OptimizeSubGraph(const SubGraph& /*subGraph*/,
                                                                       bool& optimizationAttempted) const
{
    optimizationAttempted = false;
    return nullptr;
}
ARMNN_NO_DEPRECATE_WARN_END

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

// Default implementation of OptimizeSubgraphView for backward compatibility with the old API.
// Override this method with a custom optimization implementation.
OptimizationViews IBackendInternal::OptimizeSubgraphView(const SubgraphView& subgraph) const
{
    bool optimizationAttempted = false;

    ARMNN_NO_DEPRECATE_WARN_BEGIN
    SubGraphUniquePtr optSubgraph = OptimizeSubGraph(subgraph, optimizationAttempted);
    ARMNN_NO_DEPRECATE_WARN_END

    OptimizationViews result;
    if (!optimizationAttempted)
    {
        result.AddUntouchedSubgraph(SubgraphView(subgraph));
    }
    else if (optSubgraph)
    {
        result.AddSubstitution({subgraph, SubgraphView(*optSubgraph.get())});
    }
    else
    {
        result.AddFailedSubgraph(SubgraphView(subgraph));
    }

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
