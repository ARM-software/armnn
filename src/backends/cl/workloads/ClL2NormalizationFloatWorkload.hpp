//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/backends/Workload.hpp>

#include <arm_compute/runtime/CL/functions/CLL2NormalizeLayer.h>

namespace armnn
{

arm_compute::Status ClL2NormalizationWorkloadValidate(const TensorInfo& input,
                                                      const TensorInfo& output,
                                                      const L2NormalizationDescriptor& descriptor);

class ClL2NormalizationFloatWorkload : public FloatWorkload<L2NormalizationQueueDescriptor>
{
public:
    ClL2NormalizationFloatWorkload(const L2NormalizationQueueDescriptor& descriptor,
                                   const WorkloadInfo& info,
                                   const arm_compute::CLCompileContext& clCompileContext);

    void Execute() const override;
    // Replace input tensor handle with the given TensorHandle
    void ReplaceInputTensorHandle(ITensorHandle* tensorHandle, unsigned int slot) override;

    // Replace output tensor handle with the given TensorHandle
    void ReplaceOutputTensorHandle(ITensorHandle* tensorHandle, unsigned int slot) override;

private:
    // Purposely not a CLL2Normalize function. See constructor.
    mutable arm_compute::CLL2NormalizeLayer m_Layer;
    virtual void Reconfigure();
};

} //namespace armnn




