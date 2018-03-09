//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "backends/ClWorkloadUtils.hpp"

namespace armnn
{

class ClSoftmaxFloat32Workload : public Float32Workload<SoftmaxQueueDescriptor>
{
public:
    ClSoftmaxFloat32Workload(const SoftmaxQueueDescriptor& descriptor, const WorkloadInfo& info);
    void Execute() const override;

private:
    mutable arm_compute::CLSoftmaxLayer m_SoftmaxLayer;
};

} //namespace armnn



