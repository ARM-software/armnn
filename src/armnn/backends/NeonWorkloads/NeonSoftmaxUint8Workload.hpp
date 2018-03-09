//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include <backends/NeonWorkloadUtils.hpp>

namespace armnn
{

class NeonSoftmaxUint8Workload : public Uint8Workload<SoftmaxQueueDescriptor>
{
public:
    NeonSoftmaxUint8Workload(const SoftmaxQueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    mutable arm_compute::NESoftmaxLayer m_SoftmaxLayer;
};

} //namespace armnn




