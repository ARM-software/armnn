//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <neon/workloads/NeonWorkloadUtils.hpp>

namespace armnn
{

arm_compute::Status NeonBatchNormalizationValidate(const TensorInfo& input,
                                                   const TensorInfo& output,
                                                   const TensorInfo& mean,
                                                   const TensorInfo& var,
                                                   const TensorInfo& beta,
                                                   const TensorInfo& gamma,
                                                   const BatchNormalizationDescriptor& descriptor);

class NeonBatchNormalizationFloatWorkload : public FloatWorkload<BatchNormalizationQueueDescriptor>
{
public:
    NeonBatchNormalizationFloatWorkload(const BatchNormalizationQueueDescriptor& descriptor,
                                        const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    mutable arm_compute::NEBatchNormalizationLayer m_Layer;

    std::unique_ptr<arm_compute::Tensor> m_Mean;
    std::unique_ptr<arm_compute::Tensor> m_Variance;
    std::unique_ptr<arm_compute::Tensor> m_Gamma;
    std::unique_ptr<arm_compute::Tensor> m_Beta;

    void FreeUnusedTensors();
};

} //namespace armnn



