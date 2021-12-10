//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "SampleMemoryManager.hpp"

#include <armnn/Optional.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

namespace sdb // sample dynamic backend
{

// Sample Dynamic workload factory.
class SampleDynamicWorkloadFactory : public armnn::IWorkloadFactory
{
public:
    explicit SampleDynamicWorkloadFactory(const std::shared_ptr<SampleMemoryManager>& memoryManager);
    SampleDynamicWorkloadFactory();

    ~SampleDynamicWorkloadFactory() {}

    const armnn::BackendId& GetBackendId() const override;

    static bool IsLayerSupported(const armnn::IConnectableLayer& layer,
                                 armnn::Optional<armnn::DataType> dataType,
                                 std::string& outReasonIfUnsupported);

    bool SupportsSubTensors() const override { return false; }

    ARMNN_DEPRECATED_MSG("Use ITensorHandleFactory::CreateSubTensorHandle instead")
    std::unique_ptr<armnn::ITensorHandle> CreateSubTensorHandle(
            armnn::ITensorHandle& parent,
            armnn::TensorShape const& subTensorShape,
            unsigned int const* subTensorOrigin) const override
    {
        IgnoreUnused(parent, subTensorShape, subTensorOrigin);
        return nullptr;
    }

    ARMNN_DEPRECATED_MSG("Use ITensorHandleFactory::CreateTensorHandle instead")
    std::unique_ptr<armnn::ITensorHandle> CreateTensorHandle(
            const armnn::TensorInfo& tensorInfo,
            const bool IsMemoryManaged = true) const override;

    ARMNN_DEPRECATED_MSG("Use ITensorHandleFactory::CreateTensorHandle instead")
    std::unique_ptr<armnn::ITensorHandle> CreateTensorHandle(
            const armnn::TensorInfo& tensorInfo,
            armnn::DataLayout dataLayout,
            const bool IsMemoryManaged = true) const override;

    std::unique_ptr<armnn::IWorkload> CreateAddition(
            const armnn::AdditionQueueDescriptor& descriptor,
            const armnn::WorkloadInfo& info) const override;


    std::unique_ptr<armnn::IWorkload> CreateInput(const armnn::InputQueueDescriptor& descriptor,
                                                  const armnn::WorkloadInfo& info) const override;

    std::unique_ptr<armnn::IWorkload> CreateOutput(const armnn::OutputQueueDescriptor& descriptor,
                                                   const armnn::WorkloadInfo& info) const override;

private:
    mutable std::shared_ptr<SampleMemoryManager> m_MemoryManager;

};

} // namespace sdb
