//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <arm_compute/runtime/NEON/functions/NECast.h>
#include <armnn/backends/Workload.hpp>
#include <armnn/backends/WorkloadData.hpp>
#include <memory>
#include <neon/workloads/NeonWorkloadUtils.hpp>


namespace armnn
{

arm_compute::Status NeonConvertFp16ToFp32WorkloadValidate(const TensorInfo& input, const TensorInfo& output);

class NeonConvertFp16ToFp32Workload : public Float16ToFloat32Workload<ConvertFp16ToFp32QueueDescriptor>
{
public:
    NeonConvertFp16ToFp32Workload(const ConvertFp16ToFp32QueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;
    // Replace input tensor handle with the given TensorHandle
    void ReplaceInputTensorHandle(ITensorHandle* tensorHandle, unsigned int slot) override;

    // Replace output tensor handle with the given TensorHandle
    void ReplaceOutputTensorHandle(ITensorHandle* tensorHandle, unsigned int slot) override;
private:
    using TensorHandlePair = std::pair<const ITensorHandle*, ITensorHandle*>;
    std::vector<TensorHandlePair> m_TensorHandlePairs;
    virtual void Reconfigure();
    mutable std::unique_ptr<arm_compute::NECast> m_Cast;
};

} //namespace armnn
