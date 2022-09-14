//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Types.hpp>
#include <armnn/IRuntime.hpp>
#include <armnn/Deprecated.hpp>

#include <ExecutionData.hpp>
#include <ISubgraphViewConverter.hpp>
#include <WorkingMemDescriptor.hpp>

#include <armnn/backends/IBackendContext.hpp>
#include <armnn/backends/IMemoryManager.hpp>
#include <armnn/backends/ITensorHandleFactory.hpp>
#include <armnn/backends/OptimizationViews.hpp>
#include <armnn/backends/SubgraphView.hpp>

#include <client/include/backends/IBackendProfiling.hpp>
#include <client/include/backends/IBackendProfilingContext.hpp>

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

    bool operator>=(const BackendVersion& other) const
    {
        return this->m_Major > other.m_Major ||
               (this->m_Major == other.m_Major &&
                this->m_Minor >= other.m_Minor);
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
    /// Creation must be done through a specific
    /// backend interface.
    IBackendInternal() = default;

public:
    /// Allow backends created by the factory function
    /// to be destroyed through IBackendInternal.
    ~IBackendInternal() override = default;

    using IWorkloadFactoryPtr = std::unique_ptr<IWorkloadFactory>;
    using IBackendContextPtr = std::unique_ptr<IBackendContext>;
    /// This is the bridge between backend and backend profiling we'll keep it in the backend namespace.
    using IBackendProfilingContextPtr = std::shared_ptr<arm::pipe::IBackendProfilingContext>;
    using IBackendProfilingPtr = std::unique_ptr<arm::pipe::IBackendProfiling>;
    using ILayerSupportSharedPtr = std::shared_ptr<ILayerSupport>;

    using IBackendSpecificModelContextPtr = std::shared_ptr<IBackendModelContext>;

    using IMemoryManagerUniquePtr = std::unique_ptr<IMemoryManager>;
    using IMemoryManagerSharedPtr = std::shared_ptr<IMemoryManager>;

    virtual IMemoryManagerUniquePtr CreateMemoryManager() const;

    virtual IWorkloadFactoryPtr CreateWorkloadFactory(
        const IMemoryManagerSharedPtr& memoryManager = nullptr) const = 0;

    virtual IWorkloadFactoryPtr CreateWorkloadFactory(
        class TensorHandleFactoryRegistry& tensorHandleFactoryRegistry) const;

    virtual IWorkloadFactoryPtr CreateWorkloadFactory(
        const IMemoryManagerSharedPtr& memoryManager,
        const ModelOptions& modelOptions) const;

    virtual IWorkloadFactoryPtr CreateWorkloadFactory(
        class TensorHandleFactoryRegistry& tensorHandleFactoryRegistry,
        const ModelOptions& modelOptions) const;

    virtual IWorkloadFactoryPtr CreateWorkloadFactory(
        class TensorHandleFactoryRegistry& tensorHandleFactoryRegistry,
        const ModelOptions& modelOptions,
        MemorySourceFlags inputFlags,
        MemorySourceFlags outputFlags) const;

    /// Create the runtime context of the backend
    ///
    /// Implementations may return a default-constructed IBackendContextPtr if
    /// no context is needed at runtime.
    /// Implementations must throw BackendUnavailableException if the backend
    /// cannot be used (for example, necessary accelerator hardware is not present).
    /// The default implementation always returns a default-constructed pointer.
    virtual IBackendContextPtr CreateBackendContext(const IRuntime::CreationOptions&) const;

    virtual IBackendSpecificModelContextPtr CreateBackendSpecificModelContext(const ModelOptions& modelOptions) const;

    /// Create context specifically used for profiling interaction from backends.
    virtual IBackendProfilingContextPtr CreateBackendProfilingContext(const IRuntime::CreationOptions& creationOptions,
                                                                      IBackendProfilingPtr& backendProfiling);

    virtual ILayerSupportSharedPtr GetLayerSupport() const = 0;

    virtual ILayerSupportSharedPtr GetLayerSupport(const ModelOptions& modelOptions) const;

    virtual OptimizationViews OptimizeSubgraphView(const SubgraphView& subgraph) const;

    virtual OptimizationViews OptimizeSubgraphView(const SubgraphView& subgraph,
                                                   const ModelOptions& modelOptions) const;

    bool SupportsTensorAllocatorAPI() const;

    ITensorHandleFactory::FactoryId GetBackwardCompatibleFavoriteHandleFactory();

    /// (Optional) Returns a vector of supported TensorHandleFactory ids in preference order.
    virtual std::vector<ITensorHandleFactory::FactoryId> GetHandleFactoryPreferences() const;

    /// (Optional) Register TensorHandleFactories
    /// Either this method or CreateMemoryManager() and
    /// IWorkloadFactory::CreateTensor() IWorkloadFactory::CreateSubtensor() methods must be implemented.
    virtual void RegisterTensorHandleFactories(class TensorHandleFactoryRegistry& /*registry*/) {}

    /// (Optional) Register TensorHandleFactories
    /// Either this method or CreateMemoryManager() and
    /// IWorkloadFactory::CreateTensor() IWorkloadFactory::CreateSubtensor() methods must be implemented.
    virtual void RegisterTensorHandleFactories(class TensorHandleFactoryRegistry& registry,
                                               MemorySourceFlags inputFlags,
                                               MemorySourceFlags outputFlags);

    /// Returns the version of the Backend API
    static constexpr BackendVersion GetApiVersion() { return BackendVersion(1, 0); }

    /// Returns a BackendCapability if the backend lists the capability
    /// The BackendCapability must then be inspected to check whether or not that BackendCapability is supported
    /// Otherwise returns an EmptyOptional if the BackendCapability is unlisted
    virtual BackendCapabilities GetCapabilities() const
    {
        return BackendCapabilities("IBackendInternal NullCapabilities");
    };

    /// Signals the backend to use a custom memory allocator provided by the user
    ///
    /// \param allocator - a pointer to the provided ICustomAllocator to use with this backend
    /// \param errMsg - Optional string variable to return error messages
    /// \return - Returns true if switching to custom allocator was successful
    virtual bool UseCustomMemoryAllocator(std::shared_ptr<ICustomAllocator> allocator,
                                          armnn::Optional<std::string&> errMsg)
    {
        IgnoreUnused(allocator);
        if (errMsg)
        {
            std::stringstream message;
            message << "The backend " << GetId() << " doesn't support using a custom allocator. This error might"
                                                    " be related with the protected mode if the backend doesn't"
                                                    " fully support it.";

            errMsg.value() = message.str();
        }
        return false;
    }

    /// Returns the default memory allocator for the backend
    ///
    /// \return - Returns unique pointer to the Default Allocator of the Backend
    virtual std::unique_ptr<ICustomAllocator> GetDefaultAllocator() const
    {
        throw armnn::Exception("GetDefaultAllocator: Function has not been implemented in backend.");
    }

    /// Returns the number of files cached if backend supports caching
    ///
    /// \return - Returns 0 if backend does not support caching otherwise number of files cached
    virtual unsigned int GetNumberOfCacheFiles() const { return 0; }

    /// Returns ExecutionData for the backend
    ///
    /// \param workingMemDescriptor - Vectors of input and output TensorHandles for a layer
    /// \return - Returns backend specific ExecutionData generated for a layer
    virtual ExecutionData CreateExecutionData(WorkingMemDescriptor& workingMemDescriptor) const
    {
        IgnoreUnused(workingMemDescriptor);
        throw armnn::Exception("CreateExecutionData: Function has not been implemented in backend.");
    };

    /// Update the ExecutionData for a layer. It is used to swap in pre-imported tensor handles
    ///
    /// \param executionData - Backend specific ExecutionData generated for a layer
    /// \param workingMemDescriptor - Vectors of input and output TensorHandles for a layer
    virtual void UpdateExecutionData(ExecutionData& executionData, WorkingMemDescriptor& workingMemDescriptor) const
    {
        IgnoreUnused(executionData);
        IgnoreUnused(workingMemDescriptor);
        throw armnn::Exception("UpdateExecutionData: Function has not been implemented in backend.");
    };
};

using IBackendInternalUniquePtr = std::unique_ptr<IBackendInternal>;

} // namespace armnn
