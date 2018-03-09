//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include <backends/NeonWorkloadUtils.hpp>

namespace armnn
{

class NeonBatchNormalizationFloat32Workload : public Float32Workload<BatchNormalizationQueueDescriptor>
{
public:
    NeonBatchNormalizationFloat32Workload(const BatchNormalizationQueueDescriptor& descriptor,
                                          const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    mutable arm_compute::NEBatchNormalizationLayer m_Layer;

    arm_compute::Tensor m_Mean;
    arm_compute::Tensor m_Variance;
    arm_compute::Tensor m_Gamma;
    arm_compute::Tensor m_Beta;
};

} //namespace armnn



