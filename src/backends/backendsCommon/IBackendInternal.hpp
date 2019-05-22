//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Types.hpp>
#include <armnn/IRuntime.hpp>
#include <armnn/Deprecated.hpp>

#include <ISubgraphViewConverter.hpp>
#include <SubgraphView.hpp>

#include "OptimizationViews.hpp"

#include <vector>

namespace armnn
{
class IWorkloadFactory;
class IBackendContext;
class IMemoryManager;
class Optimization;
class ILayerSupport;

class IBackendInternal : public IBackend
{
protected:
    // Creation must be done through a specific
    // backend interface.
    IBackendInternal() = default;

public:
    // Allow backends created by the factory function
    // to be destroyed through IBackendInternal.
    ~IBackendInternal() override = default;

    using IWorkloadFactoryPtr = std::unique_ptr<IWorkloadFactory>;
    using IBackendContextPtr = std::unique_ptr<IBackendContext>;
    using OptimizationPtr = std::unique_ptr<Optimization>;
    using Optimizations = std::vector<OptimizationPtr>;
    using ILayerSupportSharedPtr = std::shared_ptr<ILayerSupport>;

    using IMemoryManagerUniquePtr = std::unique_ptr<IMemoryManager>;
    using IMemoryManagerSharedPtr = std::shared_ptr<IMemoryManager>;

    using GraphUniquePtr = std::unique_ptr<Graph>;
    using SubgraphViewUniquePtr = std::unique_ptr<SubgraphView>;

    ARMNN_NO_DEPRECATE_WARN_BEGIN
    using ISubGraphConverterPtr ARMNN_DEPRECATED_MSG("This type is no longer supported")
        = std::unique_ptr<ISubGraphConverter>;
    using SubGraphUniquePtr ARMNN_DEPRECATED_MSG("SubGraph is deprecated, use SubgraphView instead")
        = std::unique_ptr<SubGraph>;

    ARMNN_DEPRECATED_MSG("This method is no longer supported")
    virtual ISubGraphConverterPtr CreateSubGraphConverter(const std::shared_ptr<SubGraph>& subGraph) const
    {
        return ISubGraphConverterPtr{};
    }

    ARMNN_DEPRECATED_MSG("Use \"OptimizationViews OptimizeSubgraphView(const SubgraphView&)\" instead")
    virtual Optimizations GetOptimizations() const = 0;

    ARMNN_DEPRECATED_MSG("Use \"OptimizationViews OptimizeSubgraphView(const SubgraphView&)\" instead")
    virtual SubGraphUniquePtr OptimizeSubGraph(const SubGraph& subGraph, bool& optimizationAttempted) const
    {
        optimizationAttempted = false;
        return nullptr;
    }
    ARMNN_NO_DEPRECATE_WARN_END

    virtual IMemoryManagerUniquePtr CreateMemoryManager() const = 0;

    virtual IWorkloadFactoryPtr CreateWorkloadFactory(
        const IMemoryManagerSharedPtr& memoryManager = nullptr) const = 0;

    virtual IBackendContextPtr CreateBackendContext(const IRuntime::CreationOptions&) const = 0;

    virtual ILayerSupportSharedPtr GetLayerSupport() const = 0;

    // Default implementation of OptimizeSubgraphView for backward compatibility with the old API.
    // Override this method with a custom optimization implementation.
    virtual OptimizationViews OptimizeSubgraphView(const SubgraphView& subgraph) const
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
        else
        {
            if (optSubgraph)
            {
                result.AddSubstitution({subgraph, SubgraphView(*optSubgraph.get())});
            }
            else
            {
                result.AddFailedSubgraph(SubgraphView(subgraph));
            }
        }
        return result;
    }
};

using IBackendInternalUniquePtr = std::unique_ptr<IBackendInternal>;

} // namespace armnn
