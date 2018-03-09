//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "backends/ClWorkloadUtils.hpp"

namespace armnn
{

class ClBatchNormalizationFloat32Workload : public Float32Workload<BatchNormalizationQueueDescriptor>
{
public:
    ClBatchNormalizationFloat32Workload(const BatchNormalizationQueueDescriptor& descriptor, const WorkloadInfo& info);

    using Float32Workload<BatchNormalizationQueueDescriptor>::Float32Workload;
    void Execute() const override;

private:
    mutable arm_compute::CLBatchNormalizationLayer m_Layer;

    arm_compute::CLTensor m_Mean;
    arm_compute::CLTensor m_Variance;
    arm_compute::CLTensor m_Gamma;
    arm_compute::CLTensor m_Beta;
};

} //namespace armnn




