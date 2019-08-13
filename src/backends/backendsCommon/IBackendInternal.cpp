//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "IBackendInternal.hpp"

namespace armnn
{

ARMNN_NO_DEPRECATE_WARN_BEGIN
IBackendInternal::ISubGraphConverterPtr IBackendInternal::CreateSubGraphConverter(
    const std::shared_ptr<SubGraph>& subGraph) const
{
    return ISubGraphConverterPtr{};
}

IBackendInternal::Optimizations IBackendInternal::GetOptimizations() const
{
    return Optimizations{};
}

IBackendInternal::SubGraphUniquePtr IBackendInternal::OptimizeSubGraph(const SubGraph& subGraph,
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
    class TensorHandleFactoryRegistry& tensorHandleFactoryRegistry) const
{
    return IWorkloadFactoryPtr{};
}

IBackendInternal::IBackendContextPtr IBackendInternal::CreateBackendContext(const IRuntime::CreationOptions&) const
{
    return IBackendContextPtr{};
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
