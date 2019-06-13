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
#include <optimizations/Optimization.hpp>

#include "IBackendContext.hpp"
#include "IMemoryManager.hpp"
#include "ITensorHandleFactory.hpp"
#include "OptimizationViews.hpp"

#include <vector>

namespace armnn
{
class IWorkloadFactory;
class IMemoryManager;
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
    virtual Optimizations GetOptimizations() const
    {
        return Optimizations{};
    }

    ARMNN_DEPRECATED_MSG("Use \"OptimizationViews OptimizeSubgraphView(const SubgraphView&)\" instead")
    virtual SubGraphUniquePtr OptimizeSubGraph(const SubGraph& subGraph, bool& optimizationAttempted) const
    {
        optimizationAttempted = false;
        return nullptr;
    }
    ARMNN_NO_DEPRECATE_WARN_END


    virtual IMemoryManagerUniquePtr CreateMemoryManager() const
    {
        return IMemoryManagerUniquePtr();
    };

    virtual IWorkloadFactoryPtr CreateWorkloadFactory(
        const IMemoryManagerSharedPtr& memoryManager = nullptr) const = 0;

    virtual IBackendContextPtr CreateBackendContext(const IRuntime::CreationOptions&) const
    {
        return IBackendContextPtr{};
    }

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

    bool SupportsTensorAllocatorAPI() const { return GetHandleFactoryPreferences().empty() == false; }

    ITensorHandleFactory::FactoryId GetBackwardCompatibleFavoriteHandleFactory()
    {
        auto favorites = GetHandleFactoryPreferences();
        if (favorites.empty())
        {
            return ITensorHandleFactory::LegacyFactoryId;
        }
        return favorites[0];
    }

    /// (Optional) Returns a vector of supported TensorHandleFactory ids in preference order.
    virtual std::vector<ITensorHandleFactory::FactoryId> GetHandleFactoryPreferences() const
    {
        return std::vector<ITensorHandleFactory::FactoryId>();
    }

    /// (Optional) Register TensorHandleFactories
    /// Either this method or CreateMemoryManager() and
    /// IWorkloadFactory::CreateTensor()/IWorkloadFactory::CreateSubtensor() methods must be implemented.
    virtual void RegisterTensorHandleFactories(class TensorHandleFactoryRegistry& registry) {}
};

using IBackendInternalUniquePtr = std::unique_ptr<IBackendInternal>;

} // namespace armnn
