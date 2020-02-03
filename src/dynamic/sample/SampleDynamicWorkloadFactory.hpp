//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "SampleMemoryManager.hpp"

#include <armnn/Optional.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

namespace armnn
{

// Sample Dynamic workload factory.
class SampleDynamicWorkloadFactory : public IWorkloadFactory
{
public:
    explicit SampleDynamicWorkloadFactory(const std::shared_ptr<SampleMemoryManager>& memoryManager);
    SampleDynamicWorkloadFactory();

    ~SampleDynamicWorkloadFactory() {}

    const BackendId& GetBackendId() const override;

    static bool IsLayerSupported(const IConnectableLayer& layer,
                                 Optional<DataType> dataType,
                                 std::string& outReasonIfUnsupported);

    bool SupportsSubTensors() const override { return false; }

    std::unique_ptr<ITensorHandle> CreateSubTensorHandle(ITensorHandle& parent,
                                                         TensorShape const& subTensorShape,
                                                         unsigned int const* subTensorOrigin) const override
    {
        boost::ignore_unused(parent, subTensorShape, subTensorOrigin);
        return nullptr;
    }

    std::unique_ptr<ITensorHandle> CreateTensorHandle(const TensorInfo& tensorInfo,
                                                      const bool IsMemoryManaged = true) const override;

    std::unique_ptr<ITensorHandle> CreateTensorHandle(const TensorInfo& tensorInfo,
                                                      DataLayout dataLayout,
                                                      const bool IsMemoryManaged = true) const override;

    std::unique_ptr<IWorkload> CreateAddition(const AdditionQueueDescriptor& descriptor,
                                              const WorkloadInfo& info) const override;


    std::unique_ptr<IWorkload> CreateInput(const InputQueueDescriptor& descriptor,
                                           const WorkloadInfo& info) const override;

    std::unique_ptr<IWorkload> CreateOutput(const OutputQueueDescriptor& descriptor,
                                            const WorkloadInfo& info) const override;

private:
    mutable std::shared_ptr<SampleMemoryManager> m_MemoryManager;

};

} // namespace armnn
