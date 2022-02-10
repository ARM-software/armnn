//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/backends/Workload.hpp>

#include <arm_compute/runtime/CL/functions/CLFloor.h>

namespace armnn
{

arm_compute::Status ClFloorWorkloadValidate(const TensorInfo& input,
                                            const TensorInfo& output);

class ClFloorFloatWorkload : public FloatWorkload<FloorQueueDescriptor>
{
public:
    ClFloorFloatWorkload(const FloorQueueDescriptor& descriptor,
                         const WorkloadInfo& info,
                         const arm_compute::CLCompileContext& clCompileContext);

    void Execute() const override;
    // Replace input tensor handle with the given TensorHandle
    void ReplaceInputTensorHandle(ITensorHandle* tensorHandle, unsigned int slot) override;

    // Replace output tensor handle with the given TensorHandle
    void ReplaceOutputTensorHandle(ITensorHandle* tensorHandle, unsigned int slot) override;
private:
    mutable arm_compute::CLFloor m_Layer;
    virtual void Reconfigure();
};

} //namespace armnn




