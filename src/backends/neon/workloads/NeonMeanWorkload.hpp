//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>

#include <arm_compute/runtime/NEON/functions/NEReduceMean.h>

namespace armnn
{

arm_compute::Status NeonMeanWorkloadValidate(const TensorInfo& input,
                                             const TensorInfo& output,
                                             const MeanDescriptor& desc);

class NeonMeanWorkload : public BaseWorkload<MeanQueueDescriptor>
{
public:
    NeonMeanWorkload(const MeanQueueDescriptor& descriptor, const WorkloadInfo& info);

    void Execute() const override;

private:
    mutable arm_compute::NEReduceMean m_Layer;
};

} //namespace armnn
