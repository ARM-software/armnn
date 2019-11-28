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
#include <memory>

namespace armnn
{
class IWorkloadFactory;
class IMemoryManager;
class ILayerSupport;

struct BackendVersion
{
    uint32_t m_Major;
    uint32_t m_Minor;

    constexpr BackendVersion()
        : m_Major(0)
        , m_Minor(0)
    {}
    constexpr BackendVersion(uint32_t major, uint32_t minor)
        : m_Major(major)
        , m_Minor(minor)
    {}

    bool operator==(const BackendVersion& other) const
    {
        return this == &other ||
               (this->m_Major == other.m_Major &&
                this->m_Minor == other.m_Minor);
    }

    bool operator<=(const BackendVersion& other) const
    {
        return this->m_Major < other.m_Major ||
               (this->m_Major == other.m_Major &&
                this->m_Minor <= other.m_Minor);
    }
};

inline std::ostream& operator<<(std::ostream& os, const BackendVersion& backendVersion)
{
    os << "[" << backendVersion.m_Major << "." << backendVersion.m_Minor << "]";

    return os;
}

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
    virtual ISubGraphConverterPtr CreateSubGraphConverter(const std::shared_ptr<SubGraph>& subGraph) const;

    ARMNN_DEPRECATED_MSG("Use \"OptimizationViews OptimizeSubgraphView(const SubgraphView&)\" instead")
    virtual Optimizations GetOptimizations() const;

    ARMNN_DEPRECATED_MSG("Use \"OptimizationViews OptimizeSubgraphView(const SubgraphView&)\" instead")
    virtual SubGraphUniquePtr OptimizeSubGraph(const SubGraph& subGraph, bool& optimizationAttempted) const;
    ARMNN_NO_DEPRECATE_WARN_END

    virtual IMemoryManagerUniquePtr CreateMemoryManager() const;

    virtual IWorkloadFactoryPtr CreateWorkloadFactory(
        const IMemoryManagerSharedPtr& memoryManager = nullptr) const = 0;

    virtual IWorkloadFactoryPtr CreateWorkloadFactory(
        class TensorHandleFactoryRegistry& tensorHandleFactoryRegistry) const;

    virtual IBackendContextPtr CreateBackendContext(const IRuntime::CreationOptions&) const;

    virtual ILayerSupportSharedPtr GetLayerSupport() const = 0;

    virtual OptimizationViews OptimizeSubgraphView(const SubgraphView& subgraph) const;

    bool SupportsTensorAllocatorAPI() const;

    ITensorHandleFactory::FactoryId GetBackwardCompatibleFavoriteHandleFactory();

    /// (Optional) Returns a vector of supported TensorHandleFactory ids in preference order.
    virtual std::vector<ITensorHandleFactory::FactoryId> GetHandleFactoryPreferences() const;

    /// (Optional) Register TensorHandleFactories
    /// Either this method or CreateMemoryManager() and
    /// IWorkloadFactory::CreateTensor()/IWorkloadFactory::CreateSubtensor() methods must be implemented.
    virtual void RegisterTensorHandleFactories(class TensorHandleFactoryRegistry& registry) {}

    /// Returns the version of the Backend API
    static constexpr BackendVersion GetApiVersion() { return BackendVersion(1, 0); }
};

using IBackendInternalUniquePtr = std::unique_ptr<IBackendInternal>;

} // namespace armnn
