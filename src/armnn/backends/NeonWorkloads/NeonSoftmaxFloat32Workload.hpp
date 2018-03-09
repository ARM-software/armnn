//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include <backends/NeonWorkloadUtils.hpp>

namespace armnn
{

class NeonSoftmaxFloat32Workload : public Float32Workload<SoftmaxQueueDescriptor>
{
public:
    NeonSoftmaxFloat32Workload(const SoftmaxQueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    mutable arm_compute::NESoftmaxLayer m_SoftmaxLayer;
};

} //namespace armnn




