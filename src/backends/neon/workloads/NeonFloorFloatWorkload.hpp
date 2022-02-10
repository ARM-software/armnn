//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/backends/Workload.hpp>

#include <arm_compute/runtime/IFunction.h>
#include <arm_compute/runtime/Tensor.h>

#include <memory>

namespace armnn
{

class NeonFloorFloatWorkload : public FloatWorkload<FloorQueueDescriptor>
{
public:
    NeonFloorFloatWorkload(const FloorQueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;
    // Replace input tensor handle with the given TensorHandle
    void ReplaceInputTensorHandle(ITensorHandle* tensorHandle, unsigned int slot) override;

    // Replace output tensor handle with the given TensorHandle
    void ReplaceOutputTensorHandle(ITensorHandle* tensorHandle, unsigned int slot) override;
private:
    std::unique_ptr<arm_compute::IFunction> m_Layer;
    virtual void Reconfigure();
};

} //namespace armnn




