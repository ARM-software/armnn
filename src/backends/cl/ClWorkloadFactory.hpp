//
// Copyright © 2017-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/IRuntime.hpp>
#include <armnn/Optional.hpp>

#include <armnn/backends/IBackendInternal.hpp>

#include <backendsCommon/WorkloadFactoryBase.hpp>
#include <aclCommon/BaseMemoryManager.hpp>

#include <arm_compute/core/CL/CLCompileContext.h>

namespace armnn
{

// ARM Compute OpenCL workload factory.
class ClWorkloadFactory : public WorkloadFactoryBase
{
public:
    ClWorkloadFactory(const std::shared_ptr<ClMemoryManager>& memoryManager);

    ClWorkloadFactory(const std::shared_ptr<ClMemoryManager>& memoryManager,
                      const IBackendInternal::IBackendSpecificModelContextPtr& modelContextPtr);

    void AfterWorkloadsCreated() override;

    const BackendId& GetBackendId() const override;

    static bool IsLayerSupported(const Layer& layer,
                                 Optional<DataType> dataType,
                                 std::string& outReasonIfUnsupported);

    static bool IsLayerSupported(const IConnectableLayer& layer,
                                 Optional<DataType> dataType,
                                 std::string& outReasonIfUnsupported,
                                 const ModelOptions& modelOptions);

    bool SupportsSubTensors() const override { return true; }

    ARMNN_DEPRECATED_MSG("Use ITensorHandleFactory::CreateSubTensorHandle instead")
    std::unique_ptr<ITensorHandle> CreateSubTensorHandle(ITensorHandle& parent,
                                                         TensorShape const& subTensorShape,
                                                         unsigned int const* subTensorOrigin) const override;

    ARMNN_DEPRECATED_MSG("Use ITensorHandleFactory::CreateTensorHandle instead")
    std::unique_ptr<ITensorHandle> CreateTensorHandle(const TensorInfo& tensorInfo,
                                                      const bool IsMemoryManaged = true) const override;

    ARMNN_DEPRECATED_MSG("Use ITensorHandleFactory::CreateTensorHandle instead")
    std::unique_ptr<ITensorHandle> CreateTensorHandle(const TensorInfo& tensorInfo,
                                                      DataLayout dataLayout,
                                                      const bool IsMemoryManaged = true) const override;
    std::unique_ptr<IWorkload> CreateWorkload(LayerType type,
                                              const QueueDescriptor& descriptor,
                                              const WorkloadInfo& info) const override;

private:
    template<typename FloatWorkload, typename Uint8Workload, typename QueueDescriptorType, typename... Args>
    static std::unique_ptr<IWorkload> MakeWorkload(const QueueDescriptorType& descriptor,
                                                   const WorkloadInfo& info,
                                                   Args&&... args);

    template <typename Workload, typename QueueDescriptorType, typename... Args>
    static std::unique_ptr<IWorkload> MakeWorkload(const QueueDescriptorType& descriptor,
                                                   const WorkloadInfo& info,
                                                   Args&&... args);

    void InitializeCLCompileContext();

    mutable std::shared_ptr<ClMemoryManager> m_MemoryManager;
    const IBackendInternal::IBackendSpecificModelContextPtr m_ModelContextPtr;
    arm_compute::CLCompileContext m_CLCompileContext;
};

} // namespace armnn
