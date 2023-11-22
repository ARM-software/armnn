//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <aclCommon/BaseMemoryManager.hpp>

#include <armnn/Optional.hpp>

namespace armnn
{

// Dynamic Fusion workload factory.
class GpuFsaWorkloadFactory : public IWorkloadFactory
{
public:
    explicit GpuFsaWorkloadFactory(const std::shared_ptr<GpuFsaMemoryManager>& memoryManager);
    GpuFsaWorkloadFactory();

    ~GpuFsaWorkloadFactory() {}

    const BackendId& GetBackendId() const override;

    static bool IsLayerSupported(const Layer& layer,
                                 Optional<DataType> dataType,
                                 std::string& outReasonIfUnsupported);

    bool SupportsSubTensors() const override { return false; }

    ARMNN_DEPRECATED_MSG("Use ITensorHandleFactory::CreateSubTensorHandle instead")
    std::unique_ptr<ITensorHandle> CreateSubTensorHandle(ITensorHandle& /*parent*/,
                                                         TensorShape const& /*subTensorShape*/,
                                                         unsigned int const* /*subTensorOrigin*/) const override
    {
        return nullptr;
    }

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
    template <typename QueueDescriptorType>
    std::unique_ptr<IWorkload> MakeWorkload(const QueueDescriptorType& descriptor, const WorkloadInfo& info) const;

    mutable std::shared_ptr<GpuFsaMemoryManager> m_MemoryManager;
};

} // namespace armnn
