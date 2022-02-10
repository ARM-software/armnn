//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClBaseWorkload.hpp"

#include <arm_compute/runtime/CL/CLTensor.h>
#include <arm_compute/runtime/CL/functions/CLBatchNormalizationLayer.h>

namespace armnn
{

arm_compute::Status ClBatchNormalizationValidate(const TensorInfo& input,
                                                 const TensorInfo& output,
                                                 const TensorInfo& mean,
                                                 const TensorInfo& var,
                                                 const TensorInfo& beta,
                                                 const TensorInfo& gamma,
                                                 const BatchNormalizationDescriptor& descriptor,
                                                 const ActivationDescriptor* activationDescriptor = nullptr);

class ClBatchNormalizationFloatWorkload : public FloatWorkload<BatchNormalizationQueueDescriptor>
{
public:
    ClBatchNormalizationFloatWorkload(const BatchNormalizationQueueDescriptor& descriptor,
                                      const WorkloadInfo& info,
                                      const arm_compute::CLCompileContext& clCompileContext);

    using FloatWorkload<BatchNormalizationQueueDescriptor>::FloatWorkload;
    void Execute() const override;

    // Replace input tensor handle with the given TensorHandle
    void ReplaceInputTensorHandle(ITensorHandle* tensorHandle, unsigned int slot) override;

    // Replace output tensor handle with the given TensorHandle
    void ReplaceOutputTensorHandle(ITensorHandle* tensorHandle, unsigned int slot) override;

private:
    mutable arm_compute::CLBatchNormalizationLayer m_Layer;

    std::unique_ptr<arm_compute::CLTensor> m_Mean;
    std::unique_ptr<arm_compute::CLTensor> m_Variance;
    std::unique_ptr<arm_compute::CLTensor> m_Gamma;
    std::unique_ptr<arm_compute::CLTensor> m_Beta;

    void FreeUnusedTensors();
    virtual void Reconfigure();
};

} //namespace armnn




