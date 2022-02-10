//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/backends/Workload.hpp>

#include <arm_compute/runtime/CL/functions/CLNormalizationLayer.h>

namespace armnn
{

arm_compute::Status ClNormalizationWorkloadValidate(const TensorInfo& input,
    const TensorInfo& output,
    const NormalizationDescriptor& descriptor);

class ClNormalizationFloatWorkload : public FloatWorkload<NormalizationQueueDescriptor>
{
public:
    ClNormalizationFloatWorkload(const NormalizationQueueDescriptor& descriptor,
                                 const WorkloadInfo& info,
                                 const arm_compute::CLCompileContext& clCompileContext);
    void Execute() const override;
    // Replace input tensor handle with the given TensorHandle
    void ReplaceInputTensorHandle(ITensorHandle* tensorHandle, unsigned int slot) override;

    // Replace output tensor handle with the given TensorHandle
    void ReplaceOutputTensorHandle(ITensorHandle* tensorHandle, unsigned int slot) override;
private:
    mutable arm_compute::CLNormalizationLayer    m_NormalizationLayer;
    virtual void Reconfigure();
};

} //namespace armnn
