//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Types.hpp>
#include <armnn/IRuntime.hpp>

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

    virtual IMemoryManagerUniquePtr CreateMemoryManager() const = 0;

    virtual IWorkloadFactoryPtr CreateWorkloadFactory(
        const IMemoryManagerSharedPtr& memoryManager = nullptr) const = 0;

    virtual IBackendContextPtr CreateBackendContext(const IRuntime::CreationOptions&) const = 0;

    virtual Optimizations GetOptimizations() const = 0;
    virtual ILayerSupportSharedPtr GetLayerSupport() const = 0;

    // @deprecated Use "OptimizationViews OptimizeSubgraphView(const SubgraphView&);" instead.
    virtual SubgraphViewUniquePtr OptimizeSubgraphView(const SubgraphView& subgraph, bool& optimizationAttempted) const
    {
        optimizationAttempted=false;
        return nullptr;
    }

    // Default implementation of OptimizeSubgraphView for backward compatibility with old API.
    // Override this method with a custom optimization implementation.
    virtual OptimizationViews OptimizeSubgraphView(const SubgraphView& subgraph) const
    {
        bool attempted=false;
        SubgraphViewUniquePtr optSubgraph = OptimizeSubgraphView(subgraph, attempted);

        OptimizationViews result;
        if (!attempted)
        {
            result.AddUntouchedSubgraph(SubgraphView(subgraph));
        }
        else
        {
            if (optSubgraph)
            {
                result.AddSubstituion({*optSubgraph.get(), subgraph});
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
