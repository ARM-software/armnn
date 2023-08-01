//
// Copyright © 2017-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Optional.hpp>
#include <armnn/backends/IBackendInternal.hpp>

#include <backendsCommon/WorkloadFactoryBase.hpp>
#include <aclCommon/BaseMemoryManager.hpp>
#include <armnn/utility/IgnoreUnused.hpp>

#include <arm_compute/runtime/IScheduler.h>

namespace armnn
{

// Neon workload factory.
class NeonWorkloadFactory : public WorkloadFactoryBase
{
public:
    NeonWorkloadFactory(const std::shared_ptr<NeonMemoryManager>& memoryManager);

    NeonWorkloadFactory(const std::shared_ptr<NeonMemoryManager>& memoryManager,
                        const IBackendInternal::IBackendSpecificModelContextPtr& modelContextPtr);

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
    void SetNumberOfThreads();

    mutable std::shared_ptr<NeonMemoryManager> m_MemoryManager;
    const IBackendInternal::IBackendSpecificModelContextPtr m_ModelContextPtr;
};

} // namespace armnn
