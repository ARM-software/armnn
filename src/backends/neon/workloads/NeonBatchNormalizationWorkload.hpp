//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "NeonBaseWorkload.hpp"

#include <arm_compute/runtime/IFunction.h>
#include <arm_compute/runtime/Tensor.h>

#include <memory>

namespace armnn
{

arm_compute::Status NeonBatchNormalizationValidate(const TensorInfo& input,
                                                   const TensorInfo& output,
                                                   const TensorInfo& mean,
                                                   const TensorInfo& var,
                                                   const TensorInfo& beta,
                                                   const TensorInfo& gamma,
                                                   const BatchNormalizationDescriptor& descriptor,
                                                   const ActivationDescriptor* activationDescriptor = nullptr);

class NeonBatchNormalizationWorkload : public NeonBaseWorkload<BatchNormalizationQueueDescriptor>
{
public:
    NeonBatchNormalizationWorkload(const BatchNormalizationQueueDescriptor& descriptor,
                                   const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    std::unique_ptr<arm_compute::IFunction> m_Layer;

    std::unique_ptr<arm_compute::Tensor> m_Mean;
    std::unique_ptr<arm_compute::Tensor> m_Variance;
    std::unique_ptr<arm_compute::Tensor> m_Gamma;
    std::unique_ptr<arm_compute::Tensor> m_Beta;

    void FreeUnusedTensors();
};

} //namespace armnn

