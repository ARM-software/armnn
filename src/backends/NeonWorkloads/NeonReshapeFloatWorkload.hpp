//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backends/NeonWorkloadUtils.hpp>

namespace armnn
{

class NeonReshapeFloatWorkload : public FloatWorkload<ReshapeQueueDescriptor>
{
public:
    NeonReshapeFloatWorkload(const ReshapeQueueDescriptor& descriptor, const WorkloadInfo& info);

    virtual void Execute() const override;

private:
    mutable arm_compute::NEReshapeLayer m_Layer;
};

} //namespace armnn





