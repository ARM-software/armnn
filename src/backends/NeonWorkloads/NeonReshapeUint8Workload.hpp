//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backends/NeonWorkloadUtils.hpp>

namespace armnn
{

class NeonReshapeUint8Workload : public Uint8Workload<ReshapeQueueDescriptor>
{
public:
    NeonReshapeUint8Workload(const ReshapeQueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    mutable arm_compute::NEReshapeLayer m_Layer;
};

} //namespace armnn




